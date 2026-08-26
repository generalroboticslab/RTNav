#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$1" == "clean" ]; then
    rm -rf build *.so
    echo "Cleaned."
    exit 0
fi

rm -rf build
mkdir -p build
cd build
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python3)}"
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE"
cmake --build . --config Release -j$(nproc)
cp -f sg_accel*.so ../
cd ..
echo "Installed to $SCRIPT_DIR"
