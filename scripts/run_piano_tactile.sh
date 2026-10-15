#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
isaac=${ISAAC_SIM_ROOT:-/isaac-sim}
python=${PIANO_TACTILE_PYTHON:-"$repo/.venv-tactile/bin/python"}
test -x "$python" || { echo 'Run scripts/install_piano_tactile.sh first.' >&2; exit 1; }
cd "$repo"
export PYTHONEXE="$python"
exec "$isaac/python.sh" -m env.tasks.run_single_finger_piano --tactile "$@"
