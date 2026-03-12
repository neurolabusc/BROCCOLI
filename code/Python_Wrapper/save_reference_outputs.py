#!/usr/bin/env python3
"""Save reference OpenCL registration outputs as NIfTI files for Stage 2 validation."""

import broccoli
import numpy as np
import scipy.io
import nibabel as nib
import os

REGISTER_DIR = '../../register'
FILTERS_DIR = '../../filters'
OUTPUT_DIR = '../../register/reference_outputs'

def load_filters():
    fp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_linear_registration.mat'))
    fnp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_nonlinear_registration.mat'))
    pf = [fp['f%d_parametric_registration' % (i+1)] for i in range(3)]
    npf = [fnp['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
    pt = [fnp['m%d' % (i+1)][0] for i in range(6)]
    fd = [fnp['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
    return pf, npf, pt, fd

def save_nifti(data, ref_img, filename):
    """Save data as NIfTI using the spatial header (sform, qform, pixdim) from ref_img."""
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(img, filename)
    print(f"  Saved {filename} {data.shape}")

def save_epi_t1():
    print("=== EPI -> T1 Registration ===")
    t1_img = nib.load(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
    (T1, T1_vs) = broccoli.load_T1(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
    (EPI, EPI_vs) = broccoli.load_EPI(os.path.join(REGISTER_DIR, 'EPI_brain.nii.gz'))
    pf, npf, pt, fd = load_filters()

    aligned, interpolated, params = broccoli.registerEPIT1(
        EPI, EPI_vs, T1, T1_vs, pf, npf, pt, fd, 20, 8, 30, 0, 0, False)

    # Output volumes are in T1 space, so use T1 header
    save_nifti(aligned, t1_img, os.path.join(OUTPUT_DIR, 'epi_t1_aligned.nii.gz'))
    save_nifti(interpolated, t1_img, os.path.join(OUTPUT_DIR, 'epi_t1_interpolated.nii.gz'))
    np.savetxt(os.path.join(OUTPUT_DIR, 'epi_t1_params.txt'), params)
    print(f"  Params: {params}")

def save_t1_mni(mni_file, label):
    print(f"=== T1 -> MNI ({label}) ===")
    mni_img = nib.load(os.path.join(REGISTER_DIR, mni_file))
    (T1, T1_vs) = broccoli.load_T1(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
    (MNI, MNI_brain, MNI_mask, MNI_vs) = broccoli.load_MNI_templates(
        os.path.join(REGISTER_DIR, mni_file))
    pf, npf, pt, fd = load_filters()

    coarsest_scale = int(round(8 / MNI_vs[0]))
    results = broccoli.registerT1MNI(T1, T1_vs, MNI, MNI_vs, MNI_brain, MNI_mask,
                                     pf, npf, pt, fd, 10, 5, coarsest_scale, 30, 0, 0, False)

    (aligned_linear, aligned_nonlinear, skullstripped, interpolated,
     params, phase_diff, phase_cert, phase_grad, slice_sums, top_slice, a_mat, h_vec,
     disp_x, disp_y, disp_z) = results

    # Output volumes are in MNI space, so use MNI header
    save_nifti(aligned_linear, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_aligned_linear.nii.gz'))
    save_nifti(aligned_nonlinear, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_aligned_nonlinear.nii.gz'))
    save_nifti(skullstripped, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_skullstripped.nii.gz'))
    save_nifti(interpolated, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_interpolated.nii.gz'))
    np.savetxt(os.path.join(OUTPUT_DIR, f't1_mni_{label}_params.txt'), params)
    print(f"  Params: {params}")

    # Save displacement fields for nonlinear diagnostic
    if disp_x is not None:
        save_nifti(disp_x, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_disp_x.nii.gz'))
        save_nifti(disp_y, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_disp_y.nii.gz'))
        save_nifti(disp_z, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_disp_z.nii.gz'))
        mag = np.sqrt(disp_x**2 + disp_y**2 + disp_z**2)
        save_nifti(mag, mni_img, os.path.join(OUTPUT_DIR, f't1_mni_{label}_disp_magnitude.nii.gz'))
        print(f"  Displacement field: min={mag.min():.2f}, max={mag.max():.2f}, mean={mag.mean():.2f}, "
              f"p99={np.percentile(mag, 99):.2f} voxels")

if __name__ == '__main__':
    import subprocess, sys
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run each registration in a separate process to avoid OpenCL context reuse issues
    tasks = [
        'save_epi_t1()',
        "save_t1_mni('MNI152_T1_2mm_brain.nii.gz', '2mm')",
        "save_t1_mni('MNI152_T1_1mm_brain.nii.gz', '1mm')",
    ]
    for task in tasks:
        code = f"import sys; sys.path.insert(0,'.'); from save_reference_outputs import *; import os; os.makedirs('{OUTPUT_DIR}', exist_ok=True); {task}"
        r = subprocess.run([sys.executable, '-c', code])
        if r.returncode != 0:
            print(f"FAILED: {task}")
            sys.exit(1)

    print("\nAll reference outputs saved to", OUTPUT_DIR)
