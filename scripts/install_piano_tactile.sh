#!/usr/bin/env bash
# Standalone Isaac Sim 5.1 / Lab 2.3.2 tactile dependencies, isolated in a venv.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix=${PIANO_TACTILE_ROOT:-"$repo/.cache/piano_tactile"}
isaac=${ISAAC_SIM_ROOT:-/isaac-sim}
python="$isaac/kit/python/bin/python3"
venv="$prefix/venvs/tactile"
vcpkg="$prefix/toolchain/vcpkg"
build="$prefix/build/tactile"
source_dir="$repo/third_party/TacEx/source"
for program in cmake ninja make g++ autoconf automake libtoolize pkg-config git curl unzip zip; do
    command -v "$program" >/dev/null || { echo "Missing build tool: $program" >&2; exit 1; }
done
test -x "$python" || { echo 'Set ISAAC_SIM_ROOT to your Isaac Sim 5.1 installation.' >&2; exit 1; }
"$python" -c 'import sys; assert sys.version_info[:2] == (3,11), "This installer requires Isaac Sim 5.1 / Python 3.11"'
test -f "$source_dir/tacex_uipc/libuipc/CMakeLists.txt" || {
    echo 'Check out the bundled third_party/TacEx source before installing.' >&2; exit 1;
}
if [ ! -x "$venv/bin/python" ]; then
    "$python" -m venv --system-site-packages "$venv"
fi
"$venv/bin/python" -m pip install 'setuptools<82' wheel 'numpy==1.26.4' \
    'pybind11==2.13.6' 'mypy==1.11.2' typing_extensions toml
export PATH="$venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
command -v nvcc >/dev/null || { echo 'Set CUDA_HOME to a CUDA 12.8 toolkit.' >&2; exit 1; }
export CUDACXX
CUDACXX=$(command -v nvcc)
export VCPKG_MAX_CONCURRENCY=${PIANO_BUILD_JOBS:-8}
export VCPKG_FORCE_SYSTEM_BINARIES=1
if [ ! -d "$vcpkg/.git" ]; then
    mkdir -p "$(dirname "$vcpkg")"
    git clone --filter=blob:none "${PIANO_GITHUB_BASE:-https://github.com}/microsoft/vcpkg.git/" "$vcpkg"
    git -C "$vcpkg" checkout --detach 20ceb8adbf0504d4b0d5e4058a32f9d4fc896391
fi
printf -v download_command '%q %q --github-prefix %q' \
    "$venv/bin/python" "$repo/scripts/_piano_tactile_download.py" "${PIANO_GITHUB_DOWNLOAD_PREFIX:-}"
export X_VCPKG_ASSET_SOURCES="clear;x-script,$download_command {url} {sha512} {dst};x-block-origin"
if [ ! -x "$vcpkg/vcpkg" ]; then
    # Use the tool version and checksum recorded in the pinned vcpkg source.
    source "$vcpkg/scripts/vcpkg-tool-metadata.txt"
    "$venv/bin/python" "$repo/scripts/_piano_tactile_download.py" \
        --github-prefix "${PIANO_GITHUB_DOWNLOAD_PREFIX:-}" \
        "https://github.com/microsoft/vcpkg-tool/releases/download/$VCPKG_TOOL_RELEASE_TAG/vcpkg-glibc" \
        "$VCPKG_GLIBC_SHA" "$vcpkg/vcpkg"
    chmod +x "$vcpkg/vcpkg"
fi
# The bundled development-mode CMake helper otherwise skips vcpkg after a
# failed first configure because its generated manifest is already present.
"$venv/bin/python" -c 'from pathlib import Path; import sys; Path(sys.argv[1]).unlink(missing_ok=True)' "$build/vcpkg.json"
cmake -S "$source_dir/tacex_uipc/libuipc" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DUIPC_PYTHON_EXECUTABLE_PATH="$venv/bin/python" \
    -DUIPC_BUILD_PYBIND=ON -DUIPC_DEV_MODE=ON -DUIPC_BUILD_GUI=OFF \
    -DUIPC_BUILD_TESTS=OFF -DUIPC_BUILD_EXAMPLES=OFF -DUIPC_BUILD_BENCHMARKS=OFF \
    -DVCPKG_OVERLAY_PORTS="$source_dir/tacex_uipc/overlay-ports" \
    -DCMAKE_CUDA_ARCHITECTURES="${PIANO_CUDA_ARCHITECTURES:-native}"
cmake --build "$build" --target pyuipc --parallel "$VCPKG_MAX_CONCURRENCY"
# A changed CUDA plugin need not relink pyuipc, so refresh the packaged plugins
# even when the binding target itself is already up to date.
"$venv/bin/python" "$source_dir/tacex_uipc/libuipc/scripts/after_build_pyuipc.py" \
    --target "$build/Release/bin/pyuipc.cpython-311-x86_64-linux-gnu.so" \
    --binary_dir "$build" --config Release --build_type Release
"$venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$build/python"
TACEX_UIPC_USE_PREBUILT=1 "$venv/bin/python" -m pip install --no-deps --no-build-isolation \
    -e "$source_dir/tacex_uipc" -e "$source_dir/tacex" -e "$source_dir/tacex_assets"
# TacEx's legacy dependency URL names a cp310 wheel; this task uses cp311/torch 2.7.
"$venv/bin/python" -m pip install --no-deps 'torch_scatter==2.1.2+pt27cu128' \
    -f 'https://data.pyg.org/whl/torch-2.7.0+cu128.html'
"$venv/bin/python" -m pip install --no-deps wildmeshing meshio svgwrite
if [ ! -e "$repo/.venv-tactile" ] && [ ! -L "$repo/.venv-tactile" ]; then
    ln -s "$venv" "$repo/.venv-tactile"
fi
echo 'Installed. Run: bash scripts/run_piano_tactile.sh --headless --video'
