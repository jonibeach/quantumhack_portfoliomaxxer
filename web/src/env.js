import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  /**
   * Specify your server-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars.
   */
  server: {
    NODE_ENV: z.enum(["development", "test", "production"]),
    // Quantum bridge.
    IBM_API_KEY: z.string().optional(),
    IBM_BACKEND: z.string().default("ibm_marrakesh"),
    // Absolute path to the Python project root (contains webapp_cli/, dqi_portfolio/).
    REPO_ROOT: z.string().optional(),
    // Python interpreter used to run the CLI (path or command on PATH).
    PYTHON_BIN: z.string().default("python3"),
    // Global rate limit + QPU budget guard for hardware runs (simulator is free).
    RATE_LIMIT_MAX: z.coerce.number().int().positive().default(10),
    RATE_LIMIT_WINDOW_MS: z.coerce.number().int().positive().default(60_000),
    QPU_MAX_RUNS: z.coerce.number().int().positive().default(80),
    QPU_MAX_INFLIGHT: z.coerce.number().int().positive().default(3),
  },

  /**
   * Specify your client-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars. To expose them to the client, prefix them with
   * `NEXT_PUBLIC_`.
   */
  client: {
    // NEXT_PUBLIC_CLIENTVAR: z.string(),
  },

  /**
   * You can't destruct `process.env` as a regular object in the Next.js edge runtimes (e.g.
   * middlewares) or client-side so we need to destruct manually.
   */
  runtimeEnv: {
    NODE_ENV: process.env.NODE_ENV,
    IBM_API_KEY: process.env.IBM_API_KEY,
    IBM_BACKEND: process.env.IBM_BACKEND,
    REPO_ROOT: process.env.REPO_ROOT,
    PYTHON_BIN: process.env.PYTHON_BIN,
    RATE_LIMIT_MAX: process.env.RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW_MS: process.env.RATE_LIMIT_WINDOW_MS,
    QPU_MAX_RUNS: process.env.QPU_MAX_RUNS,
    QPU_MAX_INFLIGHT: process.env.QPU_MAX_INFLIGHT,
    // NEXT_PUBLIC_CLIENTVAR: process.env.NEXT_PUBLIC_CLIENTVAR,
  },
  /**
   * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially
   * useful for Docker builds.
   */
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  /**
   * Makes it so that empty strings are treated as undefined. `SOME_VAR: z.string()` and
   * `SOME_VAR=''` will throw an error.
   */
  emptyStringAsUndefined: true,
});
