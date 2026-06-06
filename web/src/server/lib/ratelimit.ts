import "server-only";

import { env } from "~/env";

/**
 * In-memory GLOBAL rate limit + QPU budget guard for hardware runs.
 *
 * Correct here because Railway runs this as a single container instance. If the
 * app were ever scaled horizontally, swap these module singletons for a shared
 * store (e.g. Upstash Redis) — the call sites would not change.
 *
 * The simulator path bypasses all of this; only real QPU submissions are gated.
 */

// Global sliding window of recent QPU submission timestamps (ms).
let windowHits: number[] = [];
// Total QPU runs since process boot (hard budget cap to protect the QPU minutes).
let totalRuns = 0;
// Jobs we believe are still running on the QPU (decremented when a poll sees DONE).
const inflight = new Set<string>();

export interface RateDecision {
  ok: boolean;
  reason?: string;
  retryAfterMs?: number;
}

/** Check + reserve a QPU slot. Call once per hardware submission attempt. */
export function reserveQpuSlot(): RateDecision {
  const now = Date.now();
  windowHits = windowHits.filter((t) => now - t < env.RATE_LIMIT_WINDOW_MS);

  if (totalRuns >= env.QPU_MAX_RUNS) {
    return {
      ok: false,
      reason:
        "Global QPU run budget reached for this deployment. The free quantum-time allowance is exhausted. Try the simulator.",
    };
  }
  if (inflight.size >= env.QPU_MAX_INFLIGHT) {
    return {
      ok: false,
      reason: `Too many QPU jobs in flight (${inflight.size}/${env.QPU_MAX_INFLIGHT}). Please wait for one to finish.`,
    };
  }
  if (windowHits.length >= env.RATE_LIMIT_MAX) {
    const oldest = Math.min(...windowHits);
    return {
      ok: false,
      reason: `Rate limit: max ${env.RATE_LIMIT_MAX} QPU runs per ${Math.round(
        env.RATE_LIMIT_WINDOW_MS / 1000,
      )}s.`,
      retryAfterMs: env.RATE_LIMIT_WINDOW_MS - (now - oldest),
    };
  }

  // Reserve.
  windowHits.push(now);
  totalRuns += 1;
  return { ok: true };
}

/** Mark a submitted job as in-flight (after a successful submit). */
export function markInflight(jobId: string): void {
  inflight.add(jobId);
}

/** Release an in-flight job once a poll observes it finished. */
export function releaseInflight(jobId: string): void {
  inflight.delete(jobId);
}

/** Snapshot for display (budget banner). */
export function budgetSnapshot() {
  const now = Date.now();
  windowHits = windowHits.filter((t) => now - t < env.RATE_LIMIT_WINDOW_MS);
  return {
    windowUsed: windowHits.length,
    windowMax: env.RATE_LIMIT_MAX,
    windowMs: env.RATE_LIMIT_WINDOW_MS,
    totalRuns,
    totalMax: env.QPU_MAX_RUNS,
    inflight: inflight.size,
    inflightMax: env.QPU_MAX_INFLIGHT,
  };
}
