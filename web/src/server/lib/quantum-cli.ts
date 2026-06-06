import "server-only";

import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { env } from "~/env";

/**
 * Bridge to the Python `webapp_cli` (the only place quantum work happens). We
 * spawn `python -m webapp_cli.cli <cmd> ...` with cwd = repo root so that
 * `dqi_portfolio` and `scripts` resolve, and parse the single JSON object the
 * CLI prints to stdout.
 */

function repoRoot(): string {
  // Explicit override wins (set in the container / web/.env). Otherwise assume
  // the web app lives at <repo>/web and walk one level up.
  return env.REPO_ROOT ?? path.resolve(process.cwd(), "..");
}

/**
 * IBM credentials: prefer the process environment (Railway secrets), else fall
 * back to parsing the gitignored repo-root .env (local dev). Never logged.
 */
function ibmEnv(root: string): Record<string, string> {
  const out: Record<string, string> = {};
  const keys = ["IBM_API_KEY", "IBM_IAM_ID"] as const;
  for (const k of keys) {
    const v = process.env[k];
    if (v) out[k] = v;
  }
  if (!out.IBM_API_KEY) {
    const dotenv = path.join(root, ".env");
    if (existsSync(dotenv)) {
      try {
        for (const line of readFileSync(dotenv, "utf8").split("\n")) {
          const m = /^\s*([A-Z_]+)\s*=\s*(.*)\s*$/.exec(line);
          if (m && (keys as readonly string[]).includes(m[1]!)) {
            out[m[1]!] = m[2]!.replace(/^["']|["']$/g, "");
          }
        }
      } catch {
        // ignore — simulator path doesn't need creds
      }
    }
  }
  return out;
}

export class QuantumCliError extends Error {}

async function runCli<T>(args: string[], timeoutMs = 120_000): Promise<T> {
  const root = repoRoot();
  return new Promise<T>((resolve, reject) => {
    const child = spawn(env.PYTHON_BIN, ["-m", "webapp_cli.cli", ...args], {
      cwd: root,
      env: {
        ...process.env,
        ...ibmEnv(root),
        IBM_BACKEND: env.IBM_BACKEND,
        PYTHONUNBUFFERED: "1",
      },
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new QuantumCliError(`quantum CLI timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
    child.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(new QuantumCliError(`failed to spawn python: ${e.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      let parsed: unknown;
      try {
        parsed = JSON.parse(stdout.trim());
      } catch {
        reject(
          new QuantumCliError(
            `quantum CLI returned non-JSON (exit ${code}): ${stderr.slice(-400) || stdout.slice(-400)}`,
          ),
        );
        return;
      }
      if (parsed && typeof parsed === "object" && "error" in parsed) {
        reject(new QuantumCliError(String((parsed as { error: unknown }).error)));
        return;
      }
      resolve(parsed as T);
    });
  });
}

// ---- Shared result shapes (mirror webapp_cli/cli.py output) ----

export interface BondRow {
  bond: number;
  maturity: number;
  locator: number;
}

export interface InstanceResult {
  size: number;
  m_field: number;
  n_positions: number;
  liability_index: number;
  optimum: number;
  classical_mean_satisfied: number;
  random_mean: number;
  random_p: number;
  B: number[][];
  v: number[];
  bonds: BondRow[];
}

export interface DecodedBond {
  bond: number;
  maturity: number;
  locator: number;
}

export interface ScoreResult {
  mode: "simulator" | "hardware";
  mean: number;
  m: number;
  n_syn: number;
  optimum: number;
  p_opt: number;
  random_mean: number;
  random_p: number;
  lift: number;
  satisfied_hist: Record<string, number>;
  solution_dist: Record<string, number>;
  shots: number;
  best_solution: number[];
  best_satisfies: number;
  decoded: {
    included_bonds: number[];
    bond_details: DecodedBond[];
    residual: unknown;
  };
  size: number;
  liability_index: number;
  bonds: BondRow[];
}

export interface SubmitResult {
  job_id: string;
  backend: string;
  routed_2q: number;
  depth: number;
  shots: number;
  size: number;
  liability_index: number;
  seed: number;
}

export interface PendingPoll {
  done: false;
  status: string;
  job_id: string;
}
export interface HardwarePoll extends ScoreResult {
  done: true;
  status: string;
  job_id: string;
  backend: string | null;
}
export type PollResult = PendingPoll | HardwarePoll;

function common(size: number, liability: number, maturities?: number[]): string[] {
  const a = ["--size", String(size), "--liability", String(liability)];
  if (maturities?.length) a.push("--maturities", maturities.join(","));
  return a;
}

export const quantumCli = {
  instance: (size: number, liability: number, maturities?: number[]) =>
    runCli<InstanceResult>(["instance", ...common(size, liability, maturities)]),

  simulate: (size: number, liability: number, maturities?: number[]) =>
    runCli<ScoreResult>(["simulate", ...common(size, liability, maturities)]),

  submit: (size: number, liability: number, maturities?: number[]) =>
    runCli<SubmitResult>(["submit", ...common(size, liability, maturities)], 180_000),

  result: (jobId: string, size: number, liability: number, maturities?: number[]) =>
    runCli<PollResult>([
      "result",
      "--job-id",
      jobId,
      ...common(size, liability, maturities),
    ]),
};
