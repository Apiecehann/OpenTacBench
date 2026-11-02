#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
isaac=${ISAAC_SIM_ROOT:-/isaac-sim}
python=${PIANO_TACTILE_PYTHON:-"$repo/.venv-tactile/bin/python"}
test -x "$python" || { echo 'Run scripts/install_piano_tactile.sh first.' >&2; exit 1; }
cd "$repo"
export PYTHONEXE="$python"
torch_lib="$isaac/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib"
if [ -d "$torch_lib" ]; then
    export LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}"
fi
if [ "${1:-}" = gripper ]; then
    shift
    config=play_piano
    if [ "${1:-}" = --config ]; then
        test "$#" -ge 2 || { echo 'Missing gripper --config path.' >&2; exit 2; }
        config=$2
        shift 2
    fi
    exec "$isaac/python.sh" scripts/collect_data.py play_piano "$config" "$@"
fi
exec "$isaac/python.sh" -m env.tasks.run_single_finger_piano --tactile "$@"
