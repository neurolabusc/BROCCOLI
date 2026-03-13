# broccolini

GPU-accelerated spatial normalization (image registration) for NIfTI volumes.

Extracted from [BROCCOLI](https://github.com/wanderine/BROCCOLI) (Eklund et al., 2014) with a pluggable backend architecture supporting Metal, OpenCL, and WebGPU.

## Features

- Phase-based multi-scale registration (not mutual information)
- 6 DOF (rigid) or 12 DOF (affine) linear registration
- Optional nonlinear registration with displacement fields
- FLIRT-compatible CLI interface
- Metal backend (macOS Apple Silicon) — fast, default on macOS
- OpenCL backend — cross-platform (macOS, Linux)
- WebGPU backend — cross-platform via wgpu-native (macOS, Linux, Windows)

## Build

Requires `clang`/`clang++` and `zlib`. The Makefile auto-detects the platform.

```bash
cd src
make                    # auto-detect backend (Metal on macOS)
make BACKEND=metal      # force Metal backend
make BACKEND=opencl     # force OpenCL backend
make BACKEND=webgpu     # WebGPU backend (requires wgpu-native)
```

### Metal backend (macOS)

Requires Xcode command-line tools (provides Metal framework).

### OpenCL backend

Requires OpenCL headers and runtime. On macOS these come with Xcode; on Linux install your GPU vendor's OpenCL ICD.

The OpenCL backend also requires filter files and the `BROCCOLI_DIR` environment variable:

```bash
export BROCCOLI_DIR=$(pwd)/opencl/
```

### WebGPU backend

Requires [wgpu-native](https://github.com/gfx-rs/wgpu-native/releases) headers and library. Set `WGPU_DIR` to the extracted directory (must contain `include/` and `lib/`):

```bash
make BACKEND=webgpu WGPU_DIR=/path/to/wgpu-native
```

On macOS, ensure `DYLD_LIBRARY_PATH` includes the wgpu-native lib directory at runtime, or install the library system-wide:

```bash
DYLD_LIBRARY_PATH=/path/to/wgpu-native/lib ./broccolini ...
```

The WebGPU backend uses WGSL compute shaders (embedded in the binary) and runs on Metal (macOS), Vulkan (Linux/Windows), or DX12 (Windows) via wgpu-native.

## Usage

```
broccolini [options] -in <input> -ref <reference> -out <output>
```

### Required arguments

| Flag | Description |
|------|-------------|
| `-in <file>` | Input volume (.nii or .nii.gz) |
| `-ref <file>` | Reference/template volume |
| `-out <file>` | Output aligned volume |

### Registration options

| Flag | Default | Description |
|------|---------|-------------|
| `-dof <6\|12>` | 12 | Degrees of freedom (6=rigid, 12=affine) |
| `-lineariter <N>` | 10 | Linear iterations per scale |
| `-nonlineariter <N>` | 0 | Nonlinear iterations (0=linear only) |
| `-coarsestscale <N>` | 4 | Coarsest scale (1, 2, 4, or 8) |
| `-zcut <mm>` | 0 | Z-axis crop in mm |
| `-interp <mode>` | trilinear | nearestneighbour, trilinear, or cubic |

### I/O options

| Flag | Description |
|------|-------------|
| `-mask <file>` | Brain mask for reference (default: ref > 0) |
| `-omat <file>` | Save 4×4 affine matrix (text) |
| `-ofield <prefix>` | Save displacement field as 3 NIfTI volumes |
| `-filters <dir>` | Directory with `.bin` filter files |

### Quick start with example data

The `examples/` folder includes skull-stripped brain images for testing:

- `EPI_brain.nii.gz` — Functional (EPI) volume (64x64x33, 3mm)
- `t1_brain.nii.gz` — Structural (T1) volume (128x181x175, 1mm)
- `MNI152_T1_2mm_brain.nii.gz` — MNI template at 2mm resolution (91x109x91)
- `MNI152_T1_1mm_brain.nii.gz` — MNI template at 1mm resolution (182x218x182)

Pre-computed reference outputs are in `examples/metal/`, `examples/opencl/`, and `examples/webgpu/`.

**Run all tests for a backend:**

```bash
cd examples
bash run_tests.sh metal    # or opencl, webgpu
```

This builds broccolini, runs 5 registration tests (EPI→T1 linear, T1→MNI 2mm linear/nonlinear, T1→MNI 1mm linear/nonlinear), and saves outputs to `examples/<backend>/`.

**Individual examples:**

```bash
# EPI to T1 registration (affine)
./broccolini \
  -in examples/EPI_brain.nii.gz \
  -ref examples/t1_brain.nii.gz \
  -out /tmp/epi_t1_aligned.nii.gz \
  -omat /tmp/epi_t1_params.txt -verbose

# T1 to MNI 2mm with nonlinear registration
./broccolini \
  -in examples/t1_brain.nii.gz \
  -ref examples/MNI152_T1_2mm_brain.nii.gz \
  -out /tmp/t1_mni_2mm_nonlinear.nii.gz \
  -nonlineariter 5 -coarsestscale 4 -zcut 30 \
  -ofield /tmp/t1_mni_2mm_disp -verbose

# T1 to MNI 1mm with nonlinear registration
./broccolini \
  -in examples/t1_brain.nii.gz \
  -ref examples/MNI152_T1_1mm_brain.nii.gz \
  -out /tmp/t1_mni_1mm_nonlinear.nii.gz \
  -nonlineariter 5 -coarsestscale 8 -zcut 30 -verbose
```

**Compare outputs across backends** (run from `src/`):

```bash
python3 compare_backends.py          # compare both standalone and library outputs
python3 compare_backends.py --standalone   # standalone only
python3 compare_backends.py --library      # Python library outputs only
```

### Benchmarks

Measured on Apple M4 (macOS 15.4). Wall-clock time and peak resident memory.

| Task | Metal | OpenCL | WebGPU |
|------|-------|--------|--------|
| EPI to T1 (affine) | 0.8s / 232 MB | 1.1s / 162 MB | 1.4s / 207 MB |
| T1 to MNI 2mm (nonlinear) | 0.7s / 161 MB | 1.6s / 158 MB | 3.6s / 99 MB |
| T1 to MNI 1mm (nonlinear) | 1.9s / 798 MB | 2.6s / 234 MB | 6.4s / 411 MB |

Cross-backend NCC (normalized cross-correlation) and HF variance (mean |v[i] - v[i-1]| over consecutive non-zero voxels — higher = more preserved high-frequency detail):

| Task | Metal vs OpenCL | Metal vs WebGPU | OpenCL vs WebGPU |
|------|----------------|-----------------|------------------|
| EPI to T1 | 0.9997 | 1.0000 | 0.9997 |
| T1 to MNI 2mm nonlinear | 0.9982 | 1.0000 | 0.9982 |
| T1 to MNI 1mm nonlinear | 0.9970 | 1.0000 | 0.9970 |

| Task | Metal HF | OpenCL HF | WebGPU HF |
|------|----------|-----------|-----------|
| EPI to T1 | 26.2 | 26.3 | 26.2 |
| T1 to MNI 2mm nonlinear | 25.6 | 25.6 | 25.6 |
| T1 to MNI 1mm nonlinear | 16.9 | 16.9 | 16.9 |

## Filter files

Registration requires pre-computed quadrature filter `.bin` files. The default search path is `../filters/` relative to the executable. Override with `-filters <dir>`.

Required files (28 total):
- `filter{1-3}_{real,imag}_linear_registration.bin` — 3 linear filters
- `filter{1-6}_{real,imag}_nonlinear_registration.bin` — 6 nonlinear filters
- `projection_tensor{1-6}.bin` — 6 projection tensors
- `filter_directions_{x,y,z}.bin` — filter direction vectors

## Directory structure

```
src/
  main.c              — CLI parsing, NIfTI I/O, orchestration
  registration.h      — Backend-agnostic C API
  registration.c      — Shared utilities (filters, packing, matrix I/O)
  nifti_io.c/h        — Minimal NIfTI-1/2 reader/writer
  Makefile             — Build system
  examples/            — Test brain images (EPI, T1, MNI)
  metal/               — Metal backend (macOS)
    metal_backend.h/mm     — C vtable adapter
    metal_registration.h/mm — Metal GPU implementation
    shaders/registration.metal — Compute shaders
  opencl/              — OpenCL backend (cross-platform)
    opencl_backend.h/cpp   — C vtable adapter
    broccoli_lib.h/cpp     — BROCCOLI OpenCL implementation
    broccoli_constants.h   — Constants
    Eigen/                 — Bundled Eigen (header-only)
    kernels/               — OpenCL kernel source files
  webgpu/              — WebGPU backend (cross-platform via wgpu-native)
    webgpu_backend.h/cpp   — C vtable adapter
    webgpu_registration.h/cpp — WebGPU/WGSL GPU implementation
```

## Architecture

The executable uses a backend vtable pattern: `main.c` is pure C and calls a backend-agnostic `register_volumes()` function pointer. Each backend provides a factory function (`broc_metal_create_backend()`, `broc_opencl_create_backend()`, or `broc_webgpu_create_backend()`) that returns the vtable.

Adding a new backend (e.g. CUDA) requires:
1. Create `src/<backend>/<backend>_backend.{h,cpp}`
2. Implement the `broc_backend` vtable
3. Add `#ifdef HAVE_<BACKEND>` to `registration.h`
4. Add build rules to the Makefile

## License

See the parent repository for license information.
