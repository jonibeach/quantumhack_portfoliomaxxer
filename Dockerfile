# Single image for Railway: builds the Next.js app, then serves it from a Python
# base so the tRPC layer can spawn the webapp_cli Python bridge in-process.

# ---- Stage 1: build the Next.js web app ----
FROM node:20-bookworm-slim AS web
RUN corepack enable
WORKDIR /app/web
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
# Env is validated at runtime, not build (creds come from Railway secrets).
ENV SKIP_ENV_VALIDATION=1
RUN pnpm build

# ---- Stage 2: runtime (Python + the Node binary from the builder base) ----
FROM python:3.12-slim AS runtime
# node:20 and python:3.12-slim are both bookworm-based, so the node binary is ABI
# compatible. We only need `node` at runtime (pnpm/npm not required).
COPY --from=node:20-bookworm-slim /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# Python runtime deps (pinned to the validated local versions). The external/
# submodules are NOT needed thanks to the guarded import in dqi_portfolio/__init__.
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
      "qiskit==2.1.2" \
      "qiskit-aer==0.17.2" \
      "qiskit-ibm-runtime==0.45.0" \
      "numpy==2.4.6" \
      "scipy==1.17.1"

# Python project (imported by the CLI with cwd = /app == REPO_ROOT).
COPY pyproject.toml README.md ./
COPY dqi_portfolio/ ./dqi_portfolio/
COPY scripts/ ./scripts/
COPY webapp_cli/ ./webapp_cli/

# Built Next.js app (node_modules + .next) from stage 1.
COPY --from=web /app/web ./web

ENV NODE_ENV=production \
    REPO_ROOT=/app \
    PYTHON_BIN=python3 \
    IBM_BACKEND=ibm_marrakesh \
    PORT=3000 \
    HOSTNAME=0.0.0.0

EXPOSE 3000
WORKDIR /app/web
CMD ["node", "node_modules/next/dist/bin/next", "start", "-H", "0.0.0.0", "-p", "3000"]
