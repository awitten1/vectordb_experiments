#!/usr/bin/env bash
# Start (or stop) the Postgres built by build_postgres.sh, with pgvector enabled.
# Usage: ./run_postgres.sh [start|stop]
# Data lives in ./postgres/data; logs go to ./postgres/data/log
# Env: PGPORT (default 5432); PG_OPTS extra server options, e.g. PG_OPTS="-c shared_buffers=8GB"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="$ROOT/postgres/install/bin"
DATA="$ROOT/postgres/data"
PORT="${PGPORT:-5432}"

if [ ! -x "$BIN/pg_ctl" ]; then
  echo "error: Postgres not built yet; run ./build_postgres.sh first" >&2
  exit 1
fi

case "${1:-start}" in
  start)
    # Create the database cluster on first run (trust auth: local experiments only)
    if [ ! -d "$DATA" ]; then
      "$BIN/initdb" -D "$DATA" -U postgres -A trust
    fi
    if "$BIN/pg_ctl" -D "$DATA" status >/dev/null; then
      echo "Postgres is already running"
    else
      "$BIN/pg_ctl" -D "$DATA" -l "$DATA/log" -o "-p $PORT -c listen_addresses=localhost ${PG_OPTS:-}" -w start
    fi
    "$BIN/psql" -h localhost -p "$PORT" -U postgres -d postgres -qc "SET client_min_messages = warning; CREATE EXTENSION IF NOT EXISTS vector"
    echo "Connect with: $BIN/psql \"host=localhost port=$PORT user=postgres dbname=postgres\""
    ;;
  stop)
    if "$BIN/pg_ctl" -D "$DATA" status >/dev/null 2>&1; then
      "$BIN/pg_ctl" -D "$DATA" -w stop
    else
      echo "Postgres is not running"
    fi
    ;;
  *)
    echo "usage: $0 [start|stop]" >&2
    exit 1
    ;;
esac
