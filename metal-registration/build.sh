#!/bin/bash
# Build the Metal registration library and Python module
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Find Python and pybind11
PYTHON=${PYTHON:-python3}
PYBIND11_INCLUDES=$($PYTHON -m pybind11 --includes 2>/dev/null || echo "")
PYTHON_EXT=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null || echo ".so")

if [ -z "$PYBIND11_INCLUDES" ]; then
    echo "pybind11 not found. Install with: pip install pybind11"
    exit 1
fi

echo "=== Building Metal Registration Library ==="
echo "Python: $PYTHON"
echo "pybind11: $PYBIND11_INCLUDES"
echo "Extension: $PYTHON_EXT"

# Set the Metal shader path for runtime compilation
SHADER_PATH="$SCRIPT_DIR/src/shaders/registration.metal"

# Compile the pybind11 module
# -ObjC++ is implied by .mm extension
clang++ -std=c++17 -O2 -shared -fPIC \
    -undefined dynamic_lookup \
    -DMETAL_SHADER_DEFAULT_PATH="\"$SHADER_PATH\"" \
    $PYBIND11_INCLUDES \
    -I src \
    -framework Metal \
    -framework Foundation \
    -framework Accelerate \
    python/metal_registration_module.mm \
    src/metal_registration.mm \
    -o python/metal_registration${PYTHON_EXT}

echo "=== Build successful ==="
echo "Module: python/metal_registration${PYTHON_EXT}"
echo ""
echo "To use:"
echo "  cd python"
echo "  export METAL_SHADER_PATH=$SHADER_PATH"
echo "  python3 -c 'import metal_registration; print(\"OK\")'"
