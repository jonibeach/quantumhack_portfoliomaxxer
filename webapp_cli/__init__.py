"""Thin CLI bridge the web server shells out to.

The Next.js/tRPC layer spawns ``python -m webapp_cli.cli <cmd> ...`` (cwd = repo
root) and reads a single JSON object from stdout. This package only ORCHESTRATES
the existing quantum code in ``dqi_portfolio`` + ``scripts._immun_circuit`` — it
contains no new quantum logic.
"""
