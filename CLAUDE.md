# BROCCOLI - GPU-Accelerated fMRI Registration

## Project Overview

BROCCOLI is a GPU-accelerated fMRI analysis library (Eklund et al., 2014) using OpenCL.
Our goal is to **extract the image coregistration and reslicing functions** and build a
**Metal backend** for macOS, validated against the OpenCL reference.

## Repository Layout

```
code/
  BROCCOLI_LIB/          # Core C++ library (~21k lines)
    broccoli_lib.cpp      # Main implementation (all registration logic)
    broccoli_lib.h        # Header (BROCCOLI_LIB class, ~1500 lines)
    broccoli_constants.h  # Constants and enums
    Eigen/                # Bundled Eigen linear algebra (header-only)
    clBLASMac/            # clBLAS headers + dylib for macOS
    clBLASLinux/          # clBLAS headers + lib for Linux
  Kernels/
    kernelRegistration.cpp  # OpenCL kernels for registration (~1700 lines)
    kernelConvolution.cpp   # OpenCL kernels for convolution/filtering
    kernelMisc.cpp          # Miscellaneous kernels
    kernel*.cpp             # Other kernels (stats, clustering, etc.)
  Python_Wrapper/
    RegisterEPIT1.py        # Script: register EPI -> T1
    RegisterT1MNI.py        # Script: register T1 -> MNI
    broccoli/
      __init__.py
      broccoli_common.py    # Python BROCCOLI_LIB wrapper (pack/unpack volumes)
      registration.py       # registerEPIT1() and registerT1MNI() functions
    broccoli_lib.i          # SWIG interface file
    build.sh                # Build script for macOS (Apple Silicon)
  Bash_Wrapper/             # CLI tools (not our focus)
  Matlab_Wrapper/           # MATLAB interface (not our focus)
metal-registration/         # Metal Python library (pybind11, Stage 2)
  src/
    shaders/registration.metal  # Metal compute shaders
    metal_registration.mm       # Host code (Objective-C++)
    metal_registration.h        # Public API header
  python/
    metal_registration_module.mm  # pybind11 bindings
    validate.py                   # Validation script
  build.sh                      # Build script
webgpu-registration/        # WebGPU Python library (pure Python via wgpu-py)
  python/
    webgpu_registration.py      # WebGPU/WGSL registration (2000+ lines)
    validate.py                 # Validation script
src/                        # Standalone broccolini executable
  main.c                      # CLI parsing, NIfTI I/O, orchestration
  registration.h/c            # Backend-agnostic C API and shared utilities
  nifti_io.h/c                # Minimal NIfTI-1/2 reader/writer
  Makefile                    # Build system (auto-detects Metal/OpenCL/WebGPU)
  compare_backends.py         # Cross-backend NCC + HF variance comparison
  metal/                      # Metal backend
    metal_backend.h/mm          # C vtable adapter
    metal_registration.h/mm     # Metal GPU implementation
    shaders/registration.metal  # Compute shaders
  opencl/                     # OpenCL backend (wraps BROCCOLI_LIB)
    opencl_backend.h/cpp        # C vtable adapter
    broccoli_lib.h/cpp          # BROCCOLI OpenCL implementation
    kernels/                    # OpenCL kernel source files
  webgpu/                     # WebGPU backend (wgpu-native, WGSL embedded)
    webgpu_backend.h/cpp        # C vtable adapter
    webgpu_registration.h/cpp   # WebGPU implementation
    wgpu-native/                # Symlinks to wgpu headers + library
  examples/                   # Test data + per-backend reference outputs
    run_tests.sh                # Run all 5 tests for any backend
    metal/ opencl/ webgpu/      # Pre-computed reference outputs
filters/                    # Pre-computed quadrature filters (.bin and .mat)
register/                   # Test NIfTI images + Python library outputs
  EPI_brain.nii.gz          # Functional (EPI) test image
  t1_brain.nii.gz           # Structural (T1) test image
  MNI152_T1_2mm_brain.nii.gz  # MNI template 2mm
  MNI152_T1_1mm_brain.nii.gz  # MNI template 1mm
  reference_outputs/        # OpenCL Python library outputs
  metal_outputs/            # Metal Python library outputs
  webgpu_outputs/           # WebGPU Python library outputs
compiled/                   # Pre-compiled binaries (various platforms)
```

## Key Architecture

### Registration Pipeline
1. **Phase-based registration** using quadrature filters (not mutual information)
2. **Multi-scale** approach: coarsest to finest (scales 8 -> 4 -> 2 -> 1)
3. **Linear (affine)**: 6 params (rigid) or 12 params (full affine)
4. **Non-linear**: displacement fields via iterative deformation

### Data Flow (Python -> C++ -> OpenCL)
- Python loads NIfTI with nibabel, passes numpy arrays to SWIG-wrapped C++
- Volume packing: `flipud` + transpose `(2,0,1)` for 3D (Y,X,Z -> X,Y,Z in C)
- C++ manages OpenCL context, compiles kernels from `broccoli_lib_kernel.cpp`
- OpenCL kernels run on GPU for parallel voxel operations
- Results unpacked back to numpy arrays

### Key C++ Functions (broccoli_lib.cpp)
- `PerformRegistrationEPIT1()` / `PerformRegistrationEPIT1Wrapper()` - EPI-to-T1
- `PerformRegistrationT1MNINoSkullstrip()` / `...Wrapper()` - T1-to-MNI
- `AlignTwoVolumesLinearSeveralScales()` - Multi-scale linear registration
- `AlignTwoVolumesLinear()` - Single-scale linear registration
- `AlignTwoVolumesNonLinearSeveralScales()` - Multi-scale non-linear registration
- `TransformVolumesLinear()` / `TransformVolumesNonLinear()` - Reslicing
- `ChangeVolumesResolutionAndSize()` - Volume resampling
- `MatchVolumeMasses()` - Center-of-mass alignment

### Key OpenCL Kernels (kernelRegistration.cpp)
- `CalculatePhaseDifferencesAndCertainties` - Phase difference computation
- `CalculatePhaseGradientsX/Y/Z` - Spatial gradients
- `CalculateAMatrixAndHVector2DValuesX/Y/Z` - Least-squares system
- `CalculateAMatrix1DValues` - Reduction for solving
- `InterpolateVolume*` - Various interpolation modes

## Build Requirements (macOS)

- **Xcode**: provides g++, OpenCL framework headers
- **OpenCL headers**: `$(xcrun --show-sdk-path)/System/Library/Frameworks/OpenCL.framework/Headers/`
- **SWIG**: needed for OpenCL Python bindings (`brew install swig`)
- **pybind11**: needed for Metal Python bindings (`pip install pybind11`)
- **Python 3**: with numpy, nibabel, scipy, matplotlib
- **Eigen**: bundled in `code/BROCCOLI_LIB/Eigen/`
- **clBLAS**: bundled in `code/BROCCOLI_LIB/clBLASMac/`
- **Filter files**: `.mat` files in `filters/` (for parametric and nonparametric registration)
- **MNI templates**: Skull-stripped brain images in `register/` (no FSL needed)

## Test Datasets

All images in `register/` are skull-stripped ("brain") images. No FSL or external templates needed.

### RegisterEPIT1.py
```bash
cd code/Python_Wrapper
python RegisterEPIT1.py \
  --epi-file ../../register/EPI_brain.nii.gz \
  --t1-file ../../register/t1_brain.nii.gz \
  --filters-parametric-file ../../filters/filters_for_linear_registration.mat \
  --filters-nonparametric-file ../../filters/filters_for_nonlinear_registration.mat
```

### RegisterT1MNI.py (2mm - fast)
```bash
python RegisterT1MNI.py \
  --t1-file ../../register/t1_brain.nii.gz \
  --mni-file ../../register/MNI152_T1_2mm_brain.nii.gz \
  --filters-parametric-file ../../filters/filters_for_linear_registration.mat \
  --filters-nonparametric-file ../../filters/filters_for_nonlinear_registration.mat
```

### RegisterT1MNI.py (1mm - slower)
```bash
python RegisterT1MNI.py \
  --t1-file ../../register/t1_brain.nii.gz \
  --mni-file ../../register/MNI152_T1_1mm_brain.nii.gz \
  --filters-parametric-file ../../filters/filters_for_linear_registration.mat \
  --filters-nonparametric-file ../../filters/filters_for_nonlinear_registration.mat
```

Note: `load_MNI_templates()` uses the brain image as both the full MNI and brain template,
and derives the mask as `brain > 0`. All images in `register/` are already skull-stripped.

## Staged Development Plan

### Stage 1: Compile OpenCL Backend (COMPLETE)
**Objective**: Get the existing OpenCL Python module building and running on macOS.

**Status**: All tasks complete. Reference outputs saved in `register/reference_outputs/`.
See `register/README.md` for benchmark results and detailed documentation.

**Build**: `cd code/Python_Wrapper && bash build.sh`

**Apple Silicon kernel fixes** (in `code/Kernels/kernelRegistration.cpp`):
- `CalculateAMatricesAndHVectors`: FILTER==0 uses `=` (not `+=`) to clear stale tensor norms in d_a11
- `CalculateDisplacementUpdate`: trace-based regularization epsilon + 0.2 step-size factor
- Nonlinear registration converges for ~5-7 iterations per scale; slow drift beyond that
- Must delete `compiled/Kernels/*.bin` after kernel source changes (binary cache)

### Stage 2: Extract Registration & Metal Backend (COMPLETE)
**Objective**: Extract registration functions; create a Metal compute backend.

**Status**: All tasks complete. Metal backend passes all validation tests (NCC >= 0.85
for all three registration tasks, NCC >= 0.96 vs OpenCL reference outputs).
See `register/README.md` for benchmark results.

**Build**: `cd metal-registration && bash build.sh`

**Validate**: `cd metal-registration/python && python3 validate.py`

**Implementation** (in `metal-registration/`):
- `src/shaders/registration.metal` — Metal compute shaders (convolution, phase, tensor, interpolation)
- `src/metal_registration.mm` — Host code (device, command queue, buffer management, registration pipeline)
- `src/metal_registration.h` — Public API header
- `python/metal_registration_module.mm` — pybind11 Python bindings
- `python/validate.py` — Validation against OpenCL reference outputs

**Key differences from OpenCL**:
- Texture sampler uses `clamp_to_zero` (not `clamp_to_edge`) to avoid edge replication artifacts
- `MTLMathModeSafe` for IEEE 754 compliance (preserves NaN guards)
- Memory barriers between separable convolution passes
- `@autoreleasepool` blocks for prompt Metal buffer deallocation

### Known Issue: Z-Cut Truncates Inferior Brain in Interpolated Output

**Problem**: All backends lose ~30 inferior z-slices in the interpolated/aligned outputs.
The `MM_T1_Z_CUT` parameter (default 30mm) is applied by `CopyVolumeToNew` during
`ChangeVolumesResolutionAndSize`, which shifts the source z-index by `round(MM_Z_CUT/voxelSizeZ)`.
This is intended to exclude neck/jaw tissue from *registration*, but the truncation persists
in the output volumes because the interpolated output is read back from the already-z-cut buffer.

**Flow** (in `PerformRegistrationTwoVolumesWrapper`, `broccoli_lib.cpp:7980`):
1. `ChangeVolumesResolutionAndSize` (line 8017): resamples T1 → MNI space with z-cut applied
2. `MatchVolumeMasses` (line 8020): translates volume to align centers-of-mass
3. `h_Interpolated_T1_Volume` read back (line 8023): already has z-cut + COM shift baked in

**Texture sampler interaction** — `MatchVolumeMasses` uses `TransformVolumesLinear` which
reads from a GPU texture. The COM translation can push reads outside the volume bounds:
- **OpenCL** (`CLK_ADDRESS_CLAMP_TO_EDGE`): out-of-bounds reads repeat the edge slice,
  producing identical repeated slices in the inferior region (visible as constant-sum slices)
- **Metal** (`address::clamp_to_zero`): out-of-bounds reads return 0 (clean zeros)
- **WebGPU** (bounds-check emulation): same as Metal (zeros)

**Impact**: For `t1_crop → MNI 1mm` (where input ≈ MNI with 2 fewer y-rows), the z-cut
removes cerebellum/brainstem data unnecessarily. With 1mm isotropic voxels, slices k=0–26
are either zero (Metal/WebGPU) or repeated-edge (OpenCL), and k≥152 are zero (all backends).

**IMPORTANT for future changes**: Any new backend or refactoring must be aware that
`CopyVolumeToNew` applies `MM_Z_CUT` as a source z-offset, and all texture samplers
should use `clamp_to_zero` (not `clamp_to_edge`) to avoid edge-replication artifacts.
The OpenCL `CLK_ADDRESS_CLAMP_TO_EDGE` is a legacy bug that should eventually be fixed.

### Stage 3: Optimization & Standalone Executable (COMPLETE)
**Objective**: Optimize Metal backend; create standalone `broccolini` executable with
Metal, OpenCL, and WebGPU backends; fix high-frequency preservation across all backends.

**Metal optimizations** (in `metal-registration/`):
1. Full 3D texture convolution kernel — single dispatch replaces 7-pass z-slice loop
2. Command buffer batching — phase/grad/Amat 3-direction loop in single CB
3. Batched smoothing — multiple in-place smoothings share one CB with pre-allocated temps
4. Encoder-level helpers — fill/add/multiply encode inline instead of standalone CBs
5. Batched tensor/Amat loops in nonlinear registration

**Standalone executable** (`src/`):
- `broccolini` CLI with pluggable backend vtable (Metal, OpenCL, WebGPU)
- Build: `cd src && make BACKEND=metal` (or `opencl`, `webgpu`)
- Tests: `cd src/examples && bash run_tests.sh metal`
- Compare: `cd src && python3 compare_backends.py`

**Single-step interpolation fix** (all backends):
- OpenCL re-rescales the original T1 from scratch and applies the combined COM+affine
  (or COM+affine+nonlinear displacement) in a single interpolation step, avoiding
  compounded trilinear blur (see `broccoli_lib.cpp:8049` comment: "Do total interpolation
  in one step, to reduce smoothness").
- Metal and WebGPU originally chained 3-4 interpolation passes, losing high-frequency detail.
- Fix: after `alignTwoVolumesLinearSeveralScales`, compose COM shift into registration params
  via `composeAffineParams(regParams, initParams)`, then re-rescale original T1 and apply
  the combined transform in one step. Applied to all 4 implementations:
  - `src/metal/metal_registration.mm` (standalone Metal)
  - `src/webgpu/webgpu_registration.cpp` (standalone WebGPU)
  - `metal-registration/src/metal_registration.mm` (pybind11 Metal library)
  - `webgpu-registration/python/webgpu_registration.py` (Python WebGPU library)
- COM translation rounded to integers (`roundf`) in all backends to match OpenCL's `myround`

**Benchmarks** (Apple M4, macOS 15.4):

| Task | Metal | OpenCL | WebGPU |
|------|-------|--------|--------|
| EPI to T1 (affine) | 0.8s / 232 MB | 1.1s / 162 MB | 1.4s / 207 MB |
| T1 to MNI 2mm (nonlinear) | 0.7s / 161 MB | 1.6s / 158 MB | 3.6s / 99 MB |
| T1 to MNI 1mm (nonlinear) | 1.9s / 798 MB | 2.6s / 234 MB | 6.4s / 411 MB |

Cross-backend NCC: 0.997–1.000. HF variance matches across all backends.

**FFT convolution evaluation (USE_FFT=1)**:
- MPSGraph FFT path implemented as compile-time option (`USE_FFT=1 bash build.sh`)
- **Spatial texture3D wins at all tested sizes** — even at 58M voxels, spatial is 11.6× faster
- 7×7×7 kernel is inherently spatial-friendly

**Remaining opportunities**:
- SIMD shuffle / threadgroup memory for separable smoothing
- Memory optimization (1240 MB for 1mm — 24 filter response buffers allocated upfront)
- Unify `src/metal/` and `metal-registration/src/` to eliminate forked Metal implementation
