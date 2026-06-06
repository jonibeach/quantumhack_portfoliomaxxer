# DQI Immunization — web app

A small Next.js (T3: tRPC + Tailwind, App Router) front-end for the DQI
fixed-income immunization demo. Users search bonds (US Treasuries + a few
sovereigns, live yields via Yahoo Finance), build a maturity ladder, pick a
liability, **simulate** the algebraic-DQI result instantly, and **run it on a
real IBM quantum processor**.

## How it works

The whole quantum pipeline is Python (`qiskit` + the repo's `dqi_portfolio`
package). The tRPC server shells out to the Python bridge at the repo root:

```
browser → tRPC (web/) → spawn `python -m webapp_cli.cli <cmd> …` (cwd = repo root)
        → JSON on stdout → tRPC → React
```

The hardware circuit is the validated **t=1 syndrome-basis collapse** (3/4/5
qubits for 7/15/31 bonds), built programmatically from the instance `(B, v)` —
it amplifies ~6–16× on IBM hardware and stays under the device coherence wall.

tRPC routers (`src/server/api/routers/`):

- `bonds.search` — curated catalog + live Yahoo yields (server-side, cached).
- `immunization.preview` — AerSimulator (instant, free, unlimited).
- `quantum.run` — submit to IBM (rate-limited + budget-guarded), returns a job id.
- `quantum.poll` — poll the IBM job; interprets counts when DONE.
- `quantum.budget` — rate-limit / QPU-budget snapshot for the UI banner.

Rate limiting is **in-memory and global** (`src/server/lib/ratelimit.ts`),
correct for a single Railway instance. Simulator calls bypass all limits; only
real QPU submissions are gated.

## Local dev

```bash
cd web
pnpm install
pnpm dev            # http://localhost:3000
```

`web/.env` points the bridge at the Python project:

```
REPO_ROOT="/abs/path/to/dqi-immunization"
PYTHON_BIN="/abs/path/to/dqi-immunization/.venv/bin/python"
```

IBM credentials are read from the repo-root `.env` (`IBM_API_KEY`, gitignored)
automatically; the simulator path needs no credentials.

## Deploy (Railway)

The repo-root `Dockerfile` builds this app and serves it from a Python base so
the bridge can spawn Python in the same container. `railway.json` selects the
Dockerfile builder.

Set these Railway env vars:

| Var | Value |
|---|---|
| `IBM_API_KEY` | your IBM Quantum API key (secret) |
| `IBM_BACKEND` | `ibm_marrakesh` (default) |
| `RATE_LIMIT_MAX` | `10` (QPU runs per window) |
| `RATE_LIMIT_WINDOW_MS` | `60000` |
| `QPU_MAX_RUNS` | `80` (hard budget cap) |
| `QPU_MAX_INFLIGHT` | `3` |

`REPO_ROOT=/app` and `PYTHON_BIN=python3` are baked into the image.
