#!/usr/bin/env python3
"""Validate Metal registration against OpenCL reference outputs."""
import sys, os
import numpy as np
import nibabel as nib
import scipy.io
import time

# Add this directory to path for the metal_registration module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set shader path before importing
shader_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'src', 'shaders', 'registration.metal')
os.environ['METAL_SHADER_PATH'] = shader_path

import metal_registration

REGISTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'register')
FILTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'filters')
REF_DIR = os.path.join(REGISTER_DIR, 'reference_outputs')
METAL_OUT_DIR = os.path.join(REGISTER_DIR, 'metal_outputs')

def save_nifti(data, ref_img_path, out_path):
    """Save a volume as NIfTI using the reference image's header."""
    ref = nib.load(ref_img_path)
    img = nib.Nifti1Image(data, ref.affine, ref.header)
    nib.save(img, out_path)
    print(f"    Saved: {out_path}")

def load_volume(path):
    """Load a NIfTI volume, return (data_packed, voxel_sizes).

    Matches BROCCOLI's packVolume: flipud + transpose(2,0,1).
    After packing: shape = (nk, ni, nj) in C memory.
    C code sees W=nj, H=ni, D=nk.
    Voxel sizes swapped: X=pixdim[1], Y=pixdim[0], Z=pixdim[2].
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    # Swap voxel sizes [0] and [1] to match BROCCOLI convention
    vs = np.array([zooms[1], zooms[0], zooms[2]], dtype=np.float32)
    # Pack: flipud + transpose(2,0,1) — same as BROCCOLI's packVolume
    packed = np.ascontiguousarray(np.flipud(data).transpose(2, 0, 1))
    return packed, vs

def pack_filter(f_3d):
    """Pack a 7x7x7 filter using BROCCOLI's packVolume: flipud + transpose(2,0,1).

    Returns a flat C-order array of 343 elements matching the OpenCL layout.
    """
    return np.ascontiguousarray(np.flipud(f_3d).transpose(2, 0, 1)).flatten()

def load_filters():
    """Load quadrature filters from .mat files."""
    fp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_linear_registration.mat'))
    fnp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_nonlinear_registration.mat'))

    # Parametric filters (3, complex 7x7x7)
    # Pack real and imag separately using packVolume (same as OpenCL Python wrapper)
    pf = []
    for i in range(3):
        f = fp[f'f{i+1}_parametric_registration'].astype(np.complex64)
        fr = pack_filter(np.real(f)).astype(np.float32)
        fi = pack_filter(np.imag(f)).astype(np.float32)
        interleaved = np.column_stack([fr, fi]).astype(np.float32)
        pf.append(interleaved)

    # Nonparametric filters (6, complex 7x7x7)
    npf = []
    for i in range(6):
        f = fnp[f'f{i+1}_nonparametric_registration'].astype(np.complex64)
        fr = pack_filter(np.real(f)).astype(np.float32)
        fi = pack_filter(np.imag(f)).astype(np.float32)
        interleaved = np.column_stack([fr, fi]).astype(np.float32)
        npf.append(interleaved)

    # Projection tensors (6 tensors, each has m11,m12,m13,m22,m23,m33)
    pt = []
    for i in range(6):
        m = fnp[f'm{i+1}'][0].astype(np.float32)
        pt.append(m)

    # Filter directions
    fd = []
    for d in ['x', 'y', 'z']:
        fd.append(fnp[f'filter_directions_{d}'][0].astype(np.float32))

    return pf, npf, pt, fd

def ncc(a, b):
    """Normalized cross-correlation."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))

def unpack_volume(packed, target_shape):
    """Unpack from BROCCOLI packed format back to NIfTI (i,j,k).

    Inverse of flipud + transpose(2,0,1):
    packed shape = (nk, ni, nj) → inverse transpose(1,2,0) → (ni, nj, nk) → flipud
    BROCCOLI uses fliplr on the reshaped array, which is equivalent to flipud on the
    final (i,j,k) array since the pack did flipud on the i-axis.
    """
    # Inverse of transpose(2,0,1) is transpose(1,2,0)
    unpacked = packed.transpose(1, 2, 0)
    # Inverse of flipud
    unpacked = np.flipud(unpacked)
    return unpacked

def validate_epi_t1():
    """Validate EPI -> T1 registration."""
    print("\n" + "=" * 60)
    print("  EPI -> T1")
    print("=" * 60)

    T1, T1_vs = load_volume(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
    EPI, EPI_vs = load_volume(os.path.join(REGISTER_DIR, 'EPI_brain.nii.gz'))
    pf, npf, pt, fd = load_filters()

    t0 = time.perf_counter()
    aligned, interpolated, params = metal_registration.registerEPIT1(
        EPI, EPI_vs, T1, T1_vs, pf, npf, pt, fd, 20, 8, 30, 0, 0, False)
    dt = time.perf_counter() - t0

    # Load reference T1 for NCC
    t1_img = nib.load(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
    t1_ref = t1_img.get_fdata().astype(np.float32)

    # Unpack aligned for NCC (mask by T1 > 0, same as OpenCL Python wrapper)
    aligned_unpacked = unpack_volume(aligned, t1_ref.shape)
    t1_mask = (t1_ref > 0).astype(np.float32)
    aligned_unpacked = aligned_unpacked * t1_mask
    ncc_val = ncc(aligned_unpacked, t1_ref)

    print(f"  Time: {dt:.2f}s")
    print(f"  Output shape: {aligned.shape}")
    print(f"  NCC (aligned vs T1): {ncc_val:.4f}")
    print(f"  Parameters: {params}")

    # Compare with reference if available
    ref_path = os.path.join(REF_DIR, 'epi_t1_aligned.nii.gz')
    if os.path.exists(ref_path):
        ref_img = nib.load(ref_path)
        ref_data = ref_img.get_fdata().astype(np.float32)
        ncc_ref = ncc(aligned_unpacked, ref_data)
        print(f"  NCC (Metal vs OpenCL reference): {ncc_ref:.4f}")

    # Save outputs
    os.makedirs(METAL_OUT_DIR, exist_ok=True)
    t1_ref_path = os.path.join(REGISTER_DIR, 't1_brain.nii.gz')
    save_nifti(aligned_unpacked, t1_ref_path,
               os.path.join(METAL_OUT_DIR, 'epi_t1_aligned.nii.gz'))
    interp_unpacked = unpack_volume(interpolated, t1_ref.shape)
    save_nifti(interp_unpacked, t1_ref_path,
               os.path.join(METAL_OUT_DIR, 'epi_t1_interpolated.nii.gz'))
    np.savetxt(os.path.join(METAL_OUT_DIR, 'epi_t1_params.txt'), params)

    # Target: NCC >= 0.85 (OpenCL gets 0.900)
    status = "PASS" if ncc_val >= 0.85 else "FAIL"
    print(f"  Status: {status} (target >= 0.85, OpenCL reference: 0.900)")
    return ncc_val >= 0.85

def validate_t1_mni(res='2mm'):
    """Validate T1 -> MNI registration."""
    print(f"\n{'=' * 60}")
    print(f"  T1 -> MNI {res}")
    print("=" * 60)

    T1, T1_vs = load_volume(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
    mni_file = f'MNI152_T1_{res}_brain.nii.gz'
    mni_img = nib.load(os.path.join(REGISTER_DIR, mni_file))
    mni_data = mni_img.get_fdata().astype(np.float32)

    # Pack same as BROCCOLI: flipud + transpose(2,0,1)
    MNI = np.ascontiguousarray(np.flipud(mni_data).transpose(2, 0, 1)).astype(np.float32)
    zooms = mni_img.header.get_zooms()[:3]
    MNI_vs = np.array([zooms[1], zooms[0], zooms[2]], dtype=np.float32)

    # MNI brain = same as MNI for brain-extracted template
    MNI_brain = MNI.copy()
    # MNI mask = binary mask (nonzero voxels)
    MNI_mask = (MNI > 0).astype(np.float32)

    pf, npf, pt, fd = load_filters()

    coarsest_scale = int(round(8 / MNI_vs[0]))

    t0 = time.perf_counter()
    results = metal_registration.registerT1MNI(
        T1, T1_vs, MNI, MNI_vs, MNI_brain, MNI_mask,
        pf, npf, pt, fd, 10, 5, coarsest_scale, 30, 0, 0, False)
    dt = time.perf_counter() - t0

    al, anl, ss, interp, params, dx, dy, dz = results

    print(f"  Time: {dt:.2f}s")
    print(f"  Output shape: {al.shape}")

    for label, vol in [("linear", al), ("nonlinear", anl)]:
        vol_unpacked = unpack_volume(vol, mni_data.shape)
        ncc_val = ncc(vol_unpacked, mni_data)
        print(f"  NCC {label}: {ncc_val:.4f}")

    # Compare with references
    for label, vol, ref_name in [
        ("linear", al, f't1_mni_{res}_aligned_linear.nii.gz'),
        ("nonlinear", anl, f't1_mni_{res}_aligned_nonlinear.nii.gz'),
    ]:
        ref_path = os.path.join(REF_DIR, ref_name)
        if os.path.exists(ref_path):
            ref_data = nib.load(ref_path).get_fdata().astype(np.float32)
            vol_unpacked = unpack_volume(vol, mni_data.shape)
            ncc_ref = ncc(vol_unpacked, ref_data)
            print(f"  NCC {label} (Metal vs OpenCL ref): {ncc_ref:.4f}")

    # Save outputs
    os.makedirs(METAL_OUT_DIR, exist_ok=True)
    mni_ref_path = os.path.join(REGISTER_DIR, mni_file)
    al_unpacked = unpack_volume(al, mni_data.shape)
    anl_unpacked = unpack_volume(anl, mni_data.shape)
    save_nifti(al_unpacked, mni_ref_path,
               os.path.join(METAL_OUT_DIR, f't1_mni_{res}_aligned_linear.nii.gz'))
    save_nifti(anl_unpacked, mni_ref_path,
               os.path.join(METAL_OUT_DIR, f't1_mni_{res}_aligned_nonlinear.nii.gz'))
    ss_unpacked = unpack_volume(ss, mni_data.shape)
    save_nifti(ss_unpacked, mni_ref_path,
               os.path.join(METAL_OUT_DIR, f't1_mni_{res}_skullstripped.nii.gz'))
    interp_unpacked = unpack_volume(interp, mni_data.shape)
    save_nifti(interp_unpacked, mni_ref_path,
               os.path.join(METAL_OUT_DIR, f't1_mni_{res}_interpolated.nii.gz'))
    np.savetxt(os.path.join(METAL_OUT_DIR, f't1_mni_{res}_params.txt'), params)
    if dx.size > 0:
        for comp, name in [(dx, 'x'), (dy, 'y'), (dz, 'z')]:
            comp_unpacked = unpack_volume(comp, mni_data.shape)
            save_nifti(comp_unpacked, mni_ref_path,
                       os.path.join(METAL_OUT_DIR, f't1_mni_{res}_disp_{name}.nii.gz'))

    # Target NCCs (OpenCL: 0.931/0.934 for 2mm, 0.923/0.927 for 1mm)
    ncc_lin = ncc(al_unpacked, mni_data)
    ncc_nl = ncc(anl_unpacked, mni_data)
    target = 0.85
    status = "PASS" if ncc_nl >= target else "FAIL"
    print(f"  Status: {status} (target NCC >= {target})")
    return ncc_nl >= target

if __name__ == '__main__':
    print("Metal Registration Validation")
    print("Comparing against OpenCL reference outputs\n")

    results = []
    results.append(("EPI -> T1", validate_epi_t1()))
    results.append(("T1 -> MNI 2mm", validate_t1_mni('2mm')))
    results.append(("T1 -> MNI 1mm", validate_t1_mni('1mm')))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print("=" * 60)
    all_pass = True
    for label, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {label}: {status}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)
