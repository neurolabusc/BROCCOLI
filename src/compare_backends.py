#!/usr/bin/env python3
"""compare_backends.py — Compare registration outputs across backends.

Reports NCC (normalized cross-correlation) between all backend pairs and
a high-frequency variance metric (mean absolute difference between
consecutive non-zero voxels, via Welford's online algorithm).

Usage:
    python3 compare_backends.py                    # compare standalone outputs
    python3 compare_backends.py --library          # compare Python library outputs
    python3 compare_backends.py --all              # compare both
"""

import sys
import os
import argparse
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel required. Install with: pip install nibabel")
    sys.exit(1)


def ncc(a, b):
    """Normalized cross-correlation between two volumes."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def hf_variance(data):
    """High-frequency variance (vectorized).

    For each consecutive pair of non-zero voxels in the flattened volume,
    computes |v[i] - v[i-1]|.  Returns (mean, stddev, count).
    """
    flat = data.flatten().astype(np.float64)
    diffs = np.abs(flat[1:] - flat[:-1])
    mask = (flat[:-1] != 0.0) & (flat[1:] != 0.0)
    valid = diffs[mask]
    if len(valid) < 2:
        return 0.0, 0.0, 0
    return float(valid.mean()), float(valid.std(ddof=1)), int(len(valid))


def load_nifti(path):
    """Load a NIfTI file, return float32 data array."""
    if not os.path.exists(path):
        return None
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)


def compare_set(name, dirs, filenames):
    """Compare a set of output files across backend directories.

    dirs: dict of {backend_name: directory_path}
    filenames: list of filenames to compare
    """
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print('=' * 72)

    for fname in filenames:
        # Load all available volumes
        vols = {}
        for bname, bdir in dirs.items():
            path = os.path.join(bdir, fname)
            data = load_nifti(path)
            if data is not None:
                vols[bname] = data

        if len(vols) < 1:
            continue

        print(f"\n  {fname}")
        print(f"  {'-' * 60}")

        # HF variance for each backend
        print(f"  {'Backend':<12} {'HF mean':>10} {'HF sd':>10} {'HF pairs':>12}")
        for bname in sorted(vols.keys()):
            hf_mean, hf_sd, hf_n = hf_variance(vols[bname])
            print(f"  {bname:<12} {hf_mean:10.4f} {hf_sd:10.4f} {hf_n:12d}")

        # NCC between all pairs
        if len(vols) >= 2:
            backends = sorted(vols.keys())
            print(f"\n  {'Pair':<28} {'NCC':>10}")
            for i in range(len(backends)):
                for j in range(i + 1, len(backends)):
                    b1, b2 = backends[i], backends[j]
                    if vols[b1].shape != vols[b2].shape:
                        print(f"  {b1} vs {b2:<16} SHAPE MISMATCH "
                              f"{vols[b1].shape} vs {vols[b2].shape}")
                        continue
                    ncc_val = ncc(vols[b1], vols[b2])
                    print(f"  {b1} vs {b2:<16} {ncc_val:10.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare backend outputs")
    parser.add_argument('--library', action='store_true',
                        help='Compare Python library outputs (register/*_outputs)')
    parser.add_argument('--standalone', action='store_true',
                        help='Compare standalone C outputs (src/examples/*)')
    parser.add_argument('--all', action='store_true',
                        help='Compare both library and standalone outputs')
    args = parser.parse_args()

    # Default to --all if nothing specified
    if not args.library and not args.standalone and not args.all:
        args.all = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)

    # ---- Standalone C outputs ----
    if args.standalone or args.all:
        print("\n" + "#" * 72)
        print("#  STANDALONE C (broccolini) OUTPUTS")
        print("#" * 72)

        standalone_dirs = {}
        for backend in ['metal', 'opencl', 'webgpu']:
            d = os.path.join(script_dir, 'examples', backend)
            if os.path.isdir(d):
                standalone_dirs[backend] = d

        if standalone_dirs:
            compare_set(
                "EPI -> T1 (linear)",
                standalone_dirs,
                ['epi_t1_aligned.nii.gz'])

            compare_set(
                "T1 -> MNI 2mm (linear)",
                standalone_dirs,
                ['t1_mni_2mm_aligned_linear.nii.gz'])

            compare_set(
                "T1 -> MNI 2mm (nonlinear)",
                standalone_dirs,
                ['t1_mni_2mm_aligned_nonlinear.nii.gz'])

            compare_set(
                "T1 -> MNI 1mm (linear)",
                standalone_dirs,
                ['t1_mni_1mm_aligned_linear.nii.gz'])

            compare_set(
                "T1 -> MNI 1mm (nonlinear)",
                standalone_dirs,
                ['t1_mni_1mm_aligned_nonlinear.nii.gz'])
        else:
            print("\n  No standalone output directories found.")

    # ---- Python library outputs ----
    if args.library or args.all:
        print("\n" + "#" * 72)
        print("#  PYTHON LIBRARY OUTPUTS")
        print("#" * 72)

        register_dir = os.path.join(repo_dir, 'register')
        lib_dirs = {}
        for backend, subdir in [('opencl', 'reference_outputs'),
                                ('metal', 'metal_outputs'),
                                ('webgpu', 'webgpu_outputs')]:
            d = os.path.join(register_dir, subdir)
            if os.path.isdir(d):
                lib_dirs[backend] = d

        if lib_dirs:
            compare_set(
                "EPI -> T1 (linear)",
                lib_dirs,
                ['epi_t1_aligned.nii.gz'])

            for res in ['2mm', '1mm']:
                compare_set(
                    f"T1 -> MNI {res} (linear)",
                    lib_dirs,
                    [f't1_mni_{res}_aligned_linear.nii.gz'])

                compare_set(
                    f"T1 -> MNI {res} (nonlinear)",
                    lib_dirs,
                    [f't1_mni_{res}_aligned_nonlinear.nii.gz'])

                compare_set(
                    f"T1 -> MNI {res} (skullstripped)",
                    lib_dirs,
                    [f't1_mni_{res}_skullstripped.nii.gz'])
        else:
            print("\n  No library output directories found.")

    print()


if __name__ == '__main__':
    main()
