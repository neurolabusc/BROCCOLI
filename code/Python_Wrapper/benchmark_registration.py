#!/usr/bin/env python3
"""Benchmark registration tasks: wall time, peak memory, and NCC quality."""
import subprocess
import sys
import json

TASKS = [
    {
        "label": "EPI -> T1",
        "code": """
import broccoli, numpy as np, scipy.io, nibabel as nib, os, time, resource, json
REGISTER_DIR = '../../register'
FILTERS_DIR = '../../filters'
fp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_linear_registration.mat'))
fnp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_nonlinear_registration.mat'))
pf = [fp['f%d_parametric_registration' % (i+1)] for i in range(3)]
npf = [fnp['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
pt = [fnp['m%d' % (i+1)][0] for i in range(6)]
fd = [fnp['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
(T1, T1_vs) = broccoli.load_T1(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
(EPI, EPI_vs) = broccoli.load_EPI(os.path.join(REGISTER_DIR, 'EPI_brain.nii.gz'))
t0 = time.perf_counter()
aligned, interpolated, params = broccoli.registerEPIT1(EPI, EPI_vs, T1, T1_vs, pf, npf, pt, fd, 20, 8, 30, 0, 0, False)
dt = time.perf_counter() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
t1_img = nib.load(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
ref = t1_img.get_fdata().astype(np.float32)
a = aligned.flatten().astype(np.float64); b = ref.flatten().astype(np.float64)
a -= a.mean(); b -= b.mean()
ncc = float(np.sum(a*b) / np.sqrt(np.sum(a**2)*np.sum(b**2)))
print(json.dumps({"time_s": round(dt,2), "peak_rss_mb": round(peak_mb,1), "ncc": round(ncc,4), "output_shape": list(aligned.shape)}))
""",
    },
    {
        "label": "T1 -> MNI 2mm",
        "code": """
import broccoli, numpy as np, scipy.io, nibabel as nib, os, time, resource, json
REGISTER_DIR = '../../register'
FILTERS_DIR = '../../filters'
fp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_linear_registration.mat'))
fnp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_nonlinear_registration.mat'))
pf = [fp['f%d_parametric_registration' % (i+1)] for i in range(3)]
npf = [fnp['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
pt = [fnp['m%d' % (i+1)][0] for i in range(6)]
fd = [fnp['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
(T1, T1_vs) = broccoli.load_T1(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
mni_file = 'MNI152_T1_2mm_brain.nii.gz'
(MNI, MNI_brain, MNI_mask, MNI_vs) = broccoli.load_MNI_templates(os.path.join(REGISTER_DIR, mni_file))
mni_img = nib.load(os.path.join(REGISTER_DIR, mni_file))
mni_data = mni_img.get_fdata().astype(np.float32)
coarsest_scale = int(round(8 / MNI_vs[0]))
t0 = time.perf_counter()
results = broccoli.registerT1MNI(T1, T1_vs, MNI, MNI_vs, MNI_brain, MNI_mask, pf, npf, pt, fd, 10, 5, coarsest_scale, 30, 0, 0, False)
dt = time.perf_counter() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
(al, anl, ss, interp, params, *rest) = results
nccs = {}
for label, vol in [("linear", al), ("nonlinear", anl)]:
    a = vol.flatten().astype(np.float64); b = mni_data.flatten().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    nccs[label + "_ncc"] = round(float(np.sum(a*b) / np.sqrt(np.sum(a**2)*np.sum(b**2))), 4)
print(json.dumps({"time_s": round(dt,2), "peak_rss_mb": round(peak_mb,1), "output_shape": list(al.shape), **nccs}))
""",
    },
    {
        "label": "T1 -> MNI 1mm",
        "code": """
import broccoli, numpy as np, scipy.io, nibabel as nib, os, time, resource, json
REGISTER_DIR = '../../register'
FILTERS_DIR = '../../filters'
fp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_linear_registration.mat'))
fnp = scipy.io.loadmat(os.path.join(FILTERS_DIR, 'filters_for_nonlinear_registration.mat'))
pf = [fp['f%d_parametric_registration' % (i+1)] for i in range(3)]
npf = [fnp['f%d_nonparametric_registration' % (i+1)] for i in range(6)]
pt = [fnp['m%d' % (i+1)][0] for i in range(6)]
fd = [fnp['filter_directions_%s' % d][0] for d in ['x', 'y', 'z']]
(T1, T1_vs) = broccoli.load_T1(os.path.join(REGISTER_DIR, 't1_brain.nii.gz'))
mni_file = 'MNI152_T1_1mm_brain.nii.gz'
(MNI, MNI_brain, MNI_mask, MNI_vs) = broccoli.load_MNI_templates(os.path.join(REGISTER_DIR, mni_file))
mni_img = nib.load(os.path.join(REGISTER_DIR, mni_file))
mni_data = mni_img.get_fdata().astype(np.float32)
coarsest_scale = int(round(8 / MNI_vs[0]))
t0 = time.perf_counter()
results = broccoli.registerT1MNI(T1, T1_vs, MNI, MNI_vs, MNI_brain, MNI_mask, pf, npf, pt, fd, 10, 5, coarsest_scale, 30, 0, 0, False)
dt = time.perf_counter() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
(al, anl, ss, interp, params, *rest) = results
nccs = {}
for label, vol in [("linear", al), ("nonlinear", anl)]:
    a = vol.flatten().astype(np.float64); b = mni_data.flatten().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    nccs[label + "_ncc"] = round(float(np.sum(a*b) / np.sqrt(np.sum(a**2)*np.sum(b**2))), 4)
print(json.dumps({"time_s": round(dt,2), "peak_rss_mb": round(peak_mb,1), "output_shape": list(al.shape), **nccs}))
""",
    },
]

if __name__ == "__main__":
    results = []
    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"  {task['label']}")
        print(f"{'='*60}")
        r = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0,'.'); {task['code']}"],
            capture_output=True, text=True,
        )
        # Print OpenCL messages from stderr (suppress noise)
        for line in r.stderr.splitlines():
            if "Initializing" in line or "successful" in line or "size is" in line:
                print(line)

        info = {"label": task["label"]}
        for line in r.stdout.strip().split("\n"):
            if line.startswith("{"):
                info.update(json.loads(line))
        results.append(info)

        print(f"  Time: {info.get('time_s','?')}s  |  Peak RSS: {info.get('peak_rss_mb','?')} MB")
        if "ncc" in info:
            print(f"  NCC (aligned vs ref): {info['ncc']}")
        if "linear_ncc" in info:
            print(f"  NCC linear:    {info['linear_ncc']}")
            print(f"  NCC nonlinear: {info['nonlinear_ncc']}")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"{'Task':<20} {'Time (s)':>10} {'Peak RSS (MB)':>15} {'NCC':>10}")
    print("-" * 60)
    for r in results:
        ncc_str = str(r.get("ncc", r.get("nonlinear_ncc", "—")))
        print(f"{r['label']:<20} {r.get('time_s','?'):>10} {r.get('peak_rss_mb','?'):>15} {ncc_str:>10}")
