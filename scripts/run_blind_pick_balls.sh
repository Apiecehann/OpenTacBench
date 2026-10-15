#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
isaac=${ISAAC_SIM_ROOT:-/isaac-sim}
python=${TACTILE_PYTHON:-"$repo/.venv-tactile/bin/python"}
test -x "$python" || { echo 'Install the standalone tactile environment with scripts/install_piano_tactile.sh first.' >&2; exit 1; }
cd "$repo"
export PYTHONEXE="$python"
torch_lib="$isaac/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib"
if [ -d "$torch_lib" ]; then
    export LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}"
fi
exec "$isaac/python.sh" -m env.tasks.run_blind_pick_balls "$@"
