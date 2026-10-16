#!/usr/bin/env bash
# Isolate the original TACTO NumPy/OpenGL dependencies from Isaac/TacEx.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root="$repo/.cache/piano_tactile/digit"
python="$repo/.venv-tactile/bin/python"
test -x "$python" || { echo 'Install the Isaac tactile environment with scripts/install_piano_tactile.sh first.' >&2; exit 1; }
if [ ! -x "$root/venv/bin/python" ]; then
    "$python" -m venv "$root/venv"
fi
"$root/venv/bin/python" -m pip install \
    'numpy==1.23.5' 'scipy==1.10.1' 'trimesh==3.23.5' 'omegaconf==2.3.0' \
    'Pillow==10.4.0' 'imageio==2.37.0' 'pyglet==1.5.31' 'freetype-py==2.5.1' \
    'networkx==3.4.2' 'six==1.17.0' 'PyOpenGL==3.1.7' 'pybullet==3.2.7' 'opencv-python-headless==4.10.0.84'
# Original pyrender metadata pins an older OpenGL package; 3.1.7 provides EGL.
"$root/venv/bin/python" -m pip install --no-deps 'tacto==0.0.3' 'pyrender==0.1.45'
"$python" "$repo/scripts/_piano_digit_assets.py" "$root/assets"
