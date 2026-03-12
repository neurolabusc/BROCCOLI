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
filters/                    # Pre-computed quadrature filters (.bin and .mat)
register/                   # Test NIfTI images for validation
  EPI_brain.nii.gz          # Functional (EPI) test image
  t1_brain.nii.gz           # Structural (T1) test image
  MNI152_T1_2mm_brain.nii.gz  # MNI template 2mm
  MNI152_T1_1mm_brain.nii.gz  # MNI template 1mm
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
- **SWIG**: needed for Python bindings (`brew install swig`)
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

### Stage 2: Extract Registration & Metal Backend
**Objective**: Extract registration functions; create a Metal compute backend.

1. Identify minimal C++ subset needed for registration (separate from full BROCCOLI_LIB)
2. Create standalone registration library with clean API
3. Port OpenCL kernels to Metal Shading Language:
   - `kernelRegistration.cpp` -> Metal compute shaders
   - `kernelConvolution.cpp` (subset needed for registration filters)
4. Create Metal host code (device, command queue, buffer management)
5. Build Python bindings for Metal backend
6. Validate: compare Metal outputs to OpenCL reference (registration params, aligned volumes)

### Stage 3: Optimization
**Objective**: Optimize Metal backend for performance parity or better than OpenCL.

1. Profile Metal backend (GPU time, memory, bandwidth)
2. Optimize memory layout (simdgroup operations, threadgroup memory)
3. Optimize kernel dispatch (tile sizes, occupancy)
4. Benchmark: precision, time, memory vs OpenCL reference
5. Target: numerically equivalent results, similar memory, less wall-clock time
