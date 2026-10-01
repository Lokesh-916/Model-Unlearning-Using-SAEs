#!/usr/bin/env bash
# Launch preflight (MASTER_PLAN 4.6.1). Exit 0 only if every check passes.
#   scripts/preflight.sh [--skip-sanity]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
exec "$DSGX_PY" -m dsgx.checks.preflight "$@"
