# broccolini

GPU-accelerated spatial normalization (image registration) for NIfTI volumes.

Extracted from [BROCCOLI](https://github.com/wanderine/BROCCOLI) (Eklund et al., 2014) with a pluggable backend architecture supporting Metal and OpenCL.

## Features

- Phase-based multi-scale registration (not mutual information)
- 6 DOF (rigid) or 12 DOF (affine) linear registration
- Optional nonlinear registration with displacement fields
- FLIRT-compatible CLI interface
- Metal backend (macOS Apple Silicon) — fast, default on macOS
- OpenCL backend — cross-platform (macOS, Linux)

## Build

Requires `clang`/`clang++` and `zlib`. The Makefile auto-detects the platform.

```bash
cd src
make                    # auto-detect backend (Metal on macOS)
make BACKEND=metal      # force Metal backend
make BACKEND=opencl     # force OpenCL backend
```

### Metal backend (macOS)

Requires Xcode command-line tools (provides Metal framework).

### OpenCL backend

Requires OpenCL headers and runtime. On macOS these come with Xcode; on Linux install your GPU vendor's OpenCL ICD.

The OpenCL backend also requires filter files and the `BROCCOLI_DIR` environment variable:

```bash
export BROCCOLI_DIR=$(pwd)/opencl/
```

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

- `EPI_brain.nii.gz` — Functional (EPI) volume (80x80x32, 3mm)
- `t1_brain.nii.gz` — Structural (T1) volume (181x217x181, 1mm)
- `MNI152_T1_1mm_brain.nii.gz` — MNI template at 1mm resolution (182x218x182)

**EPI to T1 registration** (affine, ~0.2s):

```bash
./broccolini \
  -in examples/EPI_brain.nii.gz \
  -ref examples/t1_brain.nii.gz \
  -out /tmp/epi_to_t1.nii.gz \
  -dof 12 -lineariter 20 -verbose
```

**T1 to MNI spatial normalization** (affine, ~0.5s):

```bash
./broccolini \
  -in examples/t1_brain.nii.gz \
  -ref examples/MNI152_T1_1mm_brain.nii.gz \
  -out ./tmp/t1_to_mni.nii.gz
```

**T1 to MNI with nonlinear registration** (affine + warp, ~3s):

```bash
./broccolini \
  -in examples/t1_brain.nii.gz \
  -ref examples/MNI152_T1_1mm_brain.nii.gz \
  -out ./tmp/t1_to_mni_nonlinear.nii.gz \
  -nonlineariter 5 \
  -omat ./tmp/affine.txt \
  -ofield ./tmp/disp
```

This saves the aligned volume, the 4x4 affine matrix, and displacement field volumes (`disp_x.nii.gz`, `disp_y.nii.gz`, `disp_z.nii.gz`).

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
```

## Architecture

The executable uses a backend vtable pattern: `main.c` is pure C and calls a backend-agnostic `register_volumes()` function pointer. Each backend provides a factory function (`broc_metal_create_backend()` or `broc_opencl_create_backend()`) that returns the vtable.

Adding a new backend (CUDA, WebGPU, etc.) requires:
1. Create `src/<backend>/<backend>_backend.{h,cpp}`
2. Implement the `broc_backend` vtable
3. Add `#ifdef HAVE_<BACKEND>` to `registration.h`
4. Add build rules to the Makefile

## License

See the parent repository for license information.
