# Registration Reference Data and Benchmark

This directory contains input volumes and reference outputs for the BROCCOLI
GPU-accelerated image registration pipeline, running via Apple's OpenCL-to-Metal
translation layer on macOS.

## Input volumes

| File | Description | Shape | Voxel size (mm) |
|------|-------------|-------|-----------------|
| `EPI_brain.nii.gz` | Functional EPI (brain-extracted) | 64 x 64 x 33 | 3.125 x 3.125 x 3.6 |
| `t1_brain.nii.gz` | Structural T1 (brain-extracted) | 128 x 181 x 175 | 1.33 x 1.0 x 1.0 |
| `MNI152_T1_2mm_brain.nii.gz` | MNI template 2 mm | 91 x 109 x 91 | 2.0 x 2.0 x 2.0 |
| `MNI152_T1_1mm_brain.nii.gz` | MNI template 1 mm | 182 x 218 x 182 | 1.0 x 1.0 x 1.0 |

## Registration tasks

### EPI -> T1 (linear, 6-parameter rigid body)

Registers the functional EPI volume to the structural T1 using parametric
(phase-based) registration with 20 iterations across multiple scales
(coarsest scale 8, i.e. scales 8/4/2/1). Output is masked to T1 brain
extent via the OpenCL `CLK_ADDRESS_CLAMP_TO_EDGE` sampler (zero outside
the interpolation domain).

### T1 -> MNI (linear + nonlinear)

Registers the structural T1 to the MNI152 template. First performs 12-parameter
affine registration (10 iterations), then nonlinear morphon-based registration
(5 iterations per scale). The coarsest scale depends on the template voxel size:
scale 4 for 2 mm MNI (scales 4/2/1), scale 8 for 1 mm MNI (scales 8/4/2/1).

## Benchmark (Apple M4 Pro, macOS 15.3, OpenCL via Metal)

| Task | Wall time | Peak RSS | NCC (aligned vs reference) |
|------|-----------|----------|---------------------------|
| EPI -> T1 | 1.0 s | 347 MB | 0.900 |
| T1 -> MNI 2 mm | 0.8 s | 250 MB | 0.934 (linear 0.931) |
| T1 -> MNI 1 mm | 1.9 s | 822 MB | 0.927 (linear 0.923) |

NCC = normalized cross-correlation between the aligned output and the target
volume. For T1 -> MNI, both linear-only and linear+nonlinear NCC are shown.

## Benchmark (Apple M4 Pro, macOS 15.3, native Metal, optimized)

| Task | Wall time | Peak RSS | NCC (aligned vs reference) | NCC (vs OpenCL ref) |
|------|-----------|----------|---------------------------|---------------------|
| EPI -> T1 | 0.3 s | 443 MB | 0.894 | 0.962 |
| T1 -> MNI 2 mm | 0.2 s | 290 MB | 0.936 (linear 0.931) | 0.984 (linear 0.980) |
| T1 -> MNI 1 mm | 1.1 s | 1240 MB | 0.925 (linear 0.920) | 0.985 (linear 0.983) |

NCC vs OpenCL ref = normalized cross-correlation between the Metal and OpenCL
outputs for the same registration task. Wall times are roughly 3–5× faster than
OpenCL-via-Metal due to native Metal compute shaders and the following
optimizations:

1. **Full 3D texture convolution** — 7×7×7 nonseparable convolution in a single
   dispatch using `texture3d` sampling (hardware cache), replacing the 7-pass
   z-slice loop with per-pass command buffer submission.
2. **Command buffer batching** — phase/gradient/A-matrix computation for all 3
   filter directions runs in a single command buffer with memory barriers,
   reducing CPU–GPU sync points from ~16 to 2 per linear iteration.
3. **Batched smoothing** — multiple in-place Gaussian smoothings (6/9/3 per
   nonlinear iteration) share a single command buffer with pre-allocated temp
   buffers, reducing ~36 command buffers to 3.
4. **Encoder-level helpers** — utility operations (fill, add, multiply) encode
   into existing command encoders instead of creating standalone command buffers.

## Reference outputs

Saved in `reference_outputs/`:

**EPI -> T1:**
- `epi_t1_aligned.nii.gz` -- EPI aligned to T1 space (masked to T1 brain)
- `epi_t1_interpolated.nii.gz` -- EPI interpolated to T1 space (unmasked)
- `epi_t1_params.txt` -- 6 registration parameters

**T1 -> MNI (per resolution: 2mm, 1mm):**
- `t1_mni_{res}_aligned_linear.nii.gz` -- T1 after affine alignment
- `t1_mni_{res}_aligned_nonlinear.nii.gz` -- T1 after affine + nonlinear alignment
- `t1_mni_{res}_skullstripped.nii.gz` -- nonlinear result masked by MNI brain mask
- `t1_mni_{res}_interpolated.nii.gz` -- T1 resliced to MNI space (before alignment)
- `t1_mni_{res}_params.txt` -- 12 affine parameters
- `t1_mni_{res}_disp_{x,y,z}.nii.gz` -- combined displacement field (linear + nonlinear)
- `t1_mni_{res}_disp_magnitude.nii.gz` -- displacement magnitude

## Apple Silicon notes

Apple's OpenCL implementation translates kernels to Metal at runtime. Two
adjustments were required for correct nonlinear registration:

1. **d_a11 buffer reuse fix**: The `CalculateAMatricesAndHVectors` kernel uses
   assignment (`=`) for the first filter direction instead of accumulation (`+=`),
   preventing tensor-norm residuals from biasing the equation system.

2. **Regularized displacement solver**: The per-voxel 3x3 system solve uses a
   trace-based epsilon (`0.01 * trace^3 / 27`) and a step-size factor of 0.2
   to prevent overshoot from floating-point precision differences in the
   Metal translation layer.

These changes are in `code/Kernels/kernelRegistration.cpp`. The nonlinear
registration converges for approximately 5-7 iterations per scale before a slow
drift from accumulated precision errors begins to degrade alignment. The native
Metal backend in `metal-registration/` avoids this issue by using IEEE 754-safe
math mode and explicit NaN guards.

## Reproducing

```bash
cd code/Python_Wrapper
# Build the SWIG wrapper (requires Xcode CLI tools, SWIG, numpy, scipy, nibabel)
bash build.sh

# Generate reference outputs
python3 save_reference_outputs.py

# Run benchmark
python3 benchmark_registration.py
```
