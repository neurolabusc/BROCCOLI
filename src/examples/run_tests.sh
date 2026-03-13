#!/bin/bash
# run_tests.sh — Run all standalone broccolini registration tests
#
# Usage:
#   ./run_tests.sh              # auto-detect backend (Metal on macOS)
#   ./run_tests.sh metal        # force Metal backend
#   ./run_tests.sh opencl       # force OpenCL backend
#   ./run_tests.sh webgpu       # force WebGPU backend
#
# Outputs are saved to ./<backend>/ directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR"

# Determine backend
BACKEND="${1:-}"
if [ -z "$BACKEND" ]; then
    if [ "$(uname)" = "Darwin" ]; then
        BACKEND="metal"
    else
        BACKEND="opencl"
    fi
fi

MAKE_ARGS="BACKEND=$BACKEND"

# WebGPU needs WGPU_DIR for headers and library
if [ "$BACKEND" = "webgpu" ]; then
    WGPU_DIR="${WGPU_DIR:-$SRC_DIR/webgpu/wgpu-native}"
    MAKE_ARGS="$MAKE_ARGS WGPU_DIR=$WGPU_DIR"
    echo "WGPU_DIR=$WGPU_DIR"
fi

echo "=== Building broccolini (${BACKEND} backend) ==="
cd "$SRC_DIR"
make clean >/dev/null 2>&1 || true
make $MAKE_ARGS -j4
echo ""

BROCCOLINI="$SRC_DIR/broccolini"
if [ ! -x "$BROCCOLINI" ]; then
    echo "ERROR: broccolini not found at $BROCCOLINI"
    exit 1
fi

# OpenCL backend needs BROCCOLI_DIR for kernel source files
if [ "$BACKEND" = "opencl" ]; then
    export BROCCOLI_DIR="$SRC_DIR/opencl/"
    echo "BROCCOLI_DIR=$BROCCOLI_DIR"
fi

# Test data
T1="$EXAMPLES_DIR/t1_brain.nii.gz"
EPI="$EXAMPLES_DIR/EPI_brain.nii.gz"
MNI_1MM="$EXAMPLES_DIR/MNI152_T1_1mm_brain.nii.gz"
MNI_2MM="$EXAMPLES_DIR/MNI152_T1_2mm_brain.nii.gz"

# Output directory
OUT_DIR="$EXAMPLES_DIR/$BACKEND"
mkdir -p "$OUT_DIR"

echo "=== Test 1: EPI -> T1 (linear) ==="
"$BROCCOLINI" \
    -in "$EPI" -ref "$T1" \
    -out "$OUT_DIR/epi_t1_aligned.nii.gz" \
    -omat "$OUT_DIR/epi_t1_params.txt" \
    -verbose
echo ""

echo "=== Test 2: T1 -> MNI 2mm (linear) ==="
"$BROCCOLINI" \
    -in "$T1" -ref "$MNI_2MM" \
    -out "$OUT_DIR/t1_mni_2mm_aligned_linear.nii.gz" \
    -omat "$OUT_DIR/t1_mni_2mm_linear_params.txt" \
    -zcut 30 \
    -verbose
echo ""

echo "=== Test 3: T1 -> MNI 2mm (nonlinear) ==="
"$BROCCOLINI" \
    -in "$T1" -ref "$MNI_2MM" \
    -nonlineariter 5 -coarsestscale 4 \
    -out "$OUT_DIR/t1_mni_2mm_aligned_nonlinear.nii.gz" \
    -omat "$OUT_DIR/t1_mni_2mm_params.txt" \
    -ofield "$OUT_DIR/t1_mni_2mm_disp" \
    -zcut 30 \
    -verbose
echo ""

echo "=== Test 4: T1 -> MNI 1mm (linear) ==="
"$BROCCOLINI" \
    -in "$T1" -ref "$MNI_1MM" \
    -out "$OUT_DIR/t1_mni_1mm_aligned_linear.nii.gz" \
    -omat "$OUT_DIR/t1_mni_1mm_linear_params.txt" \
    -coarsestscale 8 \
    -zcut 30 \
    -verbose
echo ""

echo "=== Test 5: T1 -> MNI 1mm (nonlinear) ==="
"$BROCCOLINI" \
    -in "$T1" -ref "$MNI_1MM" \
    -nonlineariter 5 -coarsestscale 8 \
    -out "$OUT_DIR/t1_mni_1mm_aligned_nonlinear.nii.gz" \
    -omat "$OUT_DIR/t1_mni_1mm_params.txt" \
    -ofield "$OUT_DIR/t1_mni_1mm_disp" \
    -zcut 30 \
    -verbose
echo ""

echo "=== All tests complete ==="
echo "Outputs saved to: $OUT_DIR/"
ls -lh "$OUT_DIR/"
