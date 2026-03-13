#!/usr/bin/env python3
"""WebGPU-accelerated image registration (BROCCOLI backend).

Pure Python implementation using wgpu-py. Translates the Metal backend
(metal-registration/) to WebGPU/WGSL for cross-platform GPU compute.

Key differences from Metal backend:
- No hardware texture sampling; manual trilinear interpolation in shaders
- clamp_to_zero emulated via bounds checking (WebGPU lacks this address mode)
- Storage buffers everywhere (no texture3D objects)
- Each kernel is a separate WGSL module to avoid binding conflicts
"""

import numpy as np
import wgpu
import wgpu.utils
import struct
import math
import time

# ============================================================
#  WGSL Kernel Sources
# ============================================================

# Shared helper code prepended to kernels that need it
_HELPERS = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}
"""

_TRILINEAR_HELPERS = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims {
    W: i32,
    H: i32,
    D: i32,
}

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);

    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0 + 1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0 + 1, z0, W, H, D);
    let c110 = safe_read(x0 + 1, y0 + 1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0 + 1, W, H, D);
    let c101 = safe_read(x0 + 1, y0, z0 + 1, W, H, D);
    let c011 = safe_read(x0, y0 + 1, z0 + 1, W, H, D);
    let c111 = safe_read(x0 + 1, y0 + 1, z0 + 1, W, H, D);

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    return c0 * (1.0 - fz) + c1 * fz;
}
"""

# -- Utility kernels --

KERNEL_FILL_FLOAT = """
struct Params { value: f32 }
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&buf)) { return; }
    buf[i] = params.value;
}
"""

KERNEL_FILL_VEC2 = """
@group(0) @binding(0) var<storage, read_write> buf: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&buf)) { return; }
    buf[i] = vec2<f32>(0.0, 0.0);
}
"""

KERNEL_ADD_VOLUMES = """
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&A)) { return; }
    A[i] = A[i] + B[i];
}
"""

KERNEL_MULTIPLY_VOLUME = """
struct Params { factor: f32 }
@group(0) @binding(0) var<storage, read_write> vol: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&vol)) { return; }
    vol[i] = vol[i] * params.factor;
}
"""

KERNEL_MULTIPLY_VOLUMES = """
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&A)) { return; }
    A[i] = A[i] * B[i];
}
"""

# -- Reduction kernels --

KERNEL_COLUMN_MAXS = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> columnMaxs: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = i32(gid.x);
    let z = i32(gid.y);
    if (y >= dims.H || z >= dims.D) { return; }
    var mx = volume[idx3(0, y, z, dims.W, dims.H)];
    for (var x = 1; x < dims.W; x++) {
        mx = max(mx, volume[idx3(x, y, z, dims.W, dims.H)]);
    }
    columnMaxs[y + z * dims.H] = mx;
}
"""

KERNEL_ROW_MAXS = """
struct Dims { W: i32, H: i32, D: i32 }
@group(0) @binding(0) var<storage, read_write> rowMaxs: array<f32>;
@group(0) @binding(1) var<storage, read> columnMaxs: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let z = i32(gid.x);
    if (z >= dims.D) { return; }
    var mx = columnMaxs[z * dims.H];
    for (var y = 1; y < dims.H; y++) {
        mx = max(mx, columnMaxs[y + z * dims.H]);
    }
    rowMaxs[z] = mx;
}
"""

# -- 3D Nonseparable Convolution (buffer-based, 3 filters) --

KERNEL_CONV3D_FULL = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> response1: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> response2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> response3: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> volume: array<f32>;
@group(0) @binding(4) var<storage, read> f1r: array<f32>;
@group(0) @binding(5) var<storage, read> f1i: array<f32>;
@group(0) @binding(6) var<storage, read> f2r: array<f32>;
@group(0) @binding(7) var<storage, read> f2i: array<f32>;
@group(0) @binding(8) var<storage, read> f3r: array<f32>;
@group(0) @binding(9) var<storage, read> f3i: array<f32>;
@group(0) @binding(10) var<uniform> dims: Dims;

fn safe_vol(x: i32, y: i32, z: i32) -> f32 {
    if (x < 0 || x >= dims.W || y < 0 || y >= dims.H || z < 0 || z >= dims.D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, dims.W, dims.H)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var s1r: f32 = 0.0; var s1i: f32 = 0.0;
    var s2r: f32 = 0.0; var s2i: f32 = 0.0;
    var s3r: f32 = 0.0; var s3i: f32 = 0.0;

    for (var fz = 0; fz < 7; fz++) {
        for (var fy = 0; fy < 7; fy++) {
            for (var fx = 0; fx < 7; fx++) {
                let p = safe_vol(x + 3 - fx, y + 3 - fy, z + 3 - fz);
                let fi = fx + fy * 7 + fz * 49;
                s1r += f1r[fi] * p;
                s1i += f1i[fi] * p;
                s2r += f2r[fi] * p;
                s2i += f2i[fi] * p;
                s3r += f3r[fi] * p;
                s3i += f3i[fi] * p;
            }
        }
    }

    let outIdx = idx3(x, y, z, dims.W, dims.H);
    response1[outIdx] = vec2<f32>(s1r, s1i);
    response2[outIdx] = vec2<f32>(s2r, s2i);
    response3[outIdx] = vec2<f32>(s3r, s3i);
}
"""

# -- Separable convolution (smoothing) --

KERNEL_SEPARABLE_CONV_ROWS = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> filterY: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var sum: f32 = 0.0;
    for (var fy = -4; fy <= 4; fy++) {
        let yy = y + fy;
        var val: f32 = 0.0;
        if (yy >= 0 && yy < dims.H) {
            val = input[idx3(x, yy, z, dims.W, dims.H)];
        }
        sum += val * filterY[4 - fy];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}
"""

KERNEL_SEPARABLE_CONV_COLS = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> filterX: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var sum: f32 = 0.0;
    for (var fx = -4; fx <= 4; fx++) {
        let xx = x + fx;
        var val: f32 = 0.0;
        if (xx >= 0 && xx < dims.W) {
            val = input[idx3(xx, y, z, dims.W, dims.H)];
        }
        sum += val * filterX[4 - fx];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}
"""

KERNEL_SEPARABLE_CONV_RODS = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> filterZ: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    var sum: f32 = 0.0;
    for (var fz = -4; fz <= 4; fz++) {
        let zz = z + fz;
        var val: f32 = 0.0;
        if (zz >= 0 && zz < dims.D) {
            val = input[idx3(x, y, zz, dims.W, dims.H)];
        }
        sum += val * filterZ[4 - fz];
    }
    output[idx3(x, y, z, dims.W, dims.H)] = sum;
}
"""

# -- Phase differences and certainties --

KERNEL_PHASE_DIFF_CERT = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> phaseDiff: array<f32>;
@group(0) @binding(1) var<storage, read_write> certainties: array<f32>;
@group(0) @binding(2) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let a = q1[i];
    let c = q2[i];

    let cpReal = a.x * c.x + a.y * c.y;
    let cpImag = a.y * c.x - a.x * c.y;
    var phase = 0.0;
    if (abs(cpReal) > 1e-30 || abs(cpImag) > 1e-30) {
        phase = atan2(cpImag, cpReal);
    }

    let prodReal = a.x * c.x - a.y * c.y;
    let prodImag = a.y * c.x + a.x * c.y;
    let cosHalf = cos(phase * 0.5);

    phaseDiff[i] = phase;
    let mag = prodReal * prodReal + prodImag * prodImag;
    var cert = 0.0;
    if (mag > 0.0) { cert = sqrt(mag) * cosHalf * cosHalf; }
    certainties[i] = cert;
}
"""

# -- Phase gradients (X, Y, Z) --

def _make_phase_gradient_kernel(axis: str) -> str:
    """Generate phase gradient kernel for X, Y, or Z axis."""
    if axis == 'X':
        bounds_check = "if (x < 1 || x >= dims.W - 1 || y >= dims.H || z >= dims.D) { return; }"
        im_expr = "idx3(x-1, y, z, dims.W, dims.H)"
        ip_expr = "idx3(x+1, y, z, dims.W, dims.H)"
    elif axis == 'Y':
        bounds_check = "if (x >= dims.W || y < 1 || y >= dims.H - 1 || z >= dims.D) { return; }"
        im_expr = "idx3(x, y-1, z, dims.W, dims.H)"
        ip_expr = "idx3(x, y+1, z, dims.W, dims.H)"
    else:  # Z
        bounds_check = "if (x >= dims.W || y >= dims.H || z < 1 || z >= dims.D - 1) { return; }"
        im_expr = "idx3(x, y, z-1, dims.W, dims.H)"
        ip_expr = "idx3(x, y, z+1, dims.W, dims.H)"

    return _HELPERS + f"""
@group(0) @binding(0) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    {bounds_check}

    let i0 = idx3(x, y, z, dims.W, dims.H);
    let im = {im_expr};
    let ip = {ip_expr};

    var tr: f32 = 0.0; var ti: f32 = 0.0;

    // q1[ip] * conj(q1[i0])
    var a = q1[ip]; var c = q1[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q1[i0]; c = q1[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[ip]; c = q2[i0];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;
    a = q2[i0]; c = q2[im];
    tr += a.x*c.x + a.y*c.y; ti += a.y*c.x - a.x*c.y;

    var g = 0.0;
    if (abs(tr) > 1e-30 || abs(ti) > 1e-30) {{ g = atan2(ti, tr); }}
    gradients[i0] = g;
}}
"""

KERNEL_PHASE_GRAD_X = _make_phase_gradient_kernel('X')
KERNEL_PHASE_GRAD_Y = _make_phase_gradient_kernel('Y')
KERNEL_PHASE_GRAD_Z = _make_phase_gradient_kernel('Z')

# -- A-matrix and h-vector 2D --

KERNEL_AMATRIX_HVECTOR_2D = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct AMatrixParams {
    W: i32, H: i32, D: i32,
    filterSize: i32,
    directionOffset: i32,
    hVectorOffset: i32,
}

@group(0) @binding(0) var<storage, read_write> A2D: array<f32>;
@group(0) @binding(1) var<storage, read_write> h2D: array<f32>;
@group(0) @binding(2) var<storage, read> phaseDiff: array<f32>;
@group(0) @binding(3) var<storage, read> phaseGrad: array<f32>;
@group(0) @binding(4) var<storage, read> certainty: array<f32>;
@group(0) @binding(5) var<uniform> p: AMatrixParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = i32(gid.x);
    let z = i32(gid.y);
    let fhalf = (p.filterSize - 1) / 2;

    if (y < fhalf || y >= p.H - fhalf || z < fhalf || z >= p.D - fhalf) { return; }

    let yf = f32(y) - (f32(p.H) - 1.0) * 0.5;
    let zf = f32(z) - (f32(p.D) - 1.0) * 0.5;

    var aval: array<f32, 10>;
    var hval: array<f32, 4>;

    for (var x = fhalf; x < p.W - fhalf; x++) {
        let xf = f32(x) - (f32(p.W) - 1.0) * 0.5;
        let i = idx3(x, y, z, p.W, p.H);
        let pd = phaseDiff[i];
        let pg = phaseGrad[i];
        let cert = certainty[i];
        let cpp = cert * pg * pg;
        let cpd = cert * pg * pd;

        aval[0] += cpp;
        aval[1] += xf * cpp;
        aval[2] += yf * cpp;
        aval[3] += zf * cpp;
        aval[4] += xf * xf * cpp;
        aval[5] += xf * yf * cpp;
        aval[6] += xf * zf * cpp;
        aval[7] += yf * yf * cpp;
        aval[8] += yf * zf * cpp;
        aval[9] += zf * zf * cpp;

        hval[0] += cpd;
        hval[1] += xf * cpd;
        hval[2] += yf * cpd;
        hval[3] += zf * cpd;
    }

    let HD = p.H * p.D;
    let base = y + z * p.H + p.directionOffset * HD;
    for (var k = 0; k < 10; k++) {
        A2D[base + k * HD] = aval[k];
    }

    let hBase = y + z * p.H + p.hVectorOffset * HD;
    h2D[hBase] = hval[0];
    let extraBase = y + z * p.H + (3 + p.hVectorOffset * 3) * HD;
    h2D[extraBase + 0 * HD] = hval[1];
    h2D[extraBase + 1 * HD] = hval[2];
    h2D[extraBase + 2 * HD] = hval[3];
}
"""

# -- A-matrix 1D reduction --

KERNEL_AMATRIX_1D = """
@group(0) @binding(0) var<storage, read_write> A1D: array<f32>;
@group(0) @binding(1) var<storage, read> A2D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;  // H, D, filterSize, unused

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let z = i32(gid.x);
    let element = i32(gid.y);
    let H = params.x;
    let D = params.y;
    let filterSize = params.z;
    let fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf || element >= 30) { return; }

    var sum: f32 = 0.0;
    let base = z * H + element * H * D;
    for (var y = fhalf; y < H - fhalf; y++) {
        sum += A2D[base + y];
    }
    A1D[z + element * D] = sum;
}
"""

# -- A-matrix final reduction --

KERNEL_AMATRIX_FINAL = """
const parameterIndices = array<vec2<i32>, 30>(
    vec2<i32>(0,0), vec2<i32>(3,0), vec2<i32>(4,0), vec2<i32>(5,0),
    vec2<i32>(3,3), vec2<i32>(4,3), vec2<i32>(5,3), vec2<i32>(4,4),
    vec2<i32>(5,4), vec2<i32>(5,5),
    vec2<i32>(1,1), vec2<i32>(6,1), vec2<i32>(7,1), vec2<i32>(8,1),
    vec2<i32>(6,6), vec2<i32>(7,6), vec2<i32>(8,6), vec2<i32>(7,7),
    vec2<i32>(8,7), vec2<i32>(8,8),
    vec2<i32>(2,2), vec2<i32>(9,2), vec2<i32>(10,2), vec2<i32>(11,2),
    vec2<i32>(9,9), vec2<i32>(10,9), vec2<i32>(11,9), vec2<i32>(10,10),
    vec2<i32>(11,10), vec2<i32>(11,11)
);

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> A1D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;  // D, filterSize, unused, unused

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = i32(gid.x);
    if (element >= 30) { return; }
    let D = params.x;
    let filterSize = params.y;
    let fhalf = (filterSize - 1) / 2;

    var sum: f32 = 0.0;
    let base = element * D;
    for (var z = fhalf; z < D - fhalf; z++) {
        sum += A1D[base + z];
    }

    let ij = parameterIndices[element];
    A[ij.x + ij.y * 12] = sum;
}
"""

# -- H-vector 1D reduction --

KERNEL_HVECTOR_1D = """
@group(0) @binding(0) var<storage, read_write> h1D: array<f32>;
@group(0) @binding(1) var<storage, read> h2D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;  // H, D, filterSize, unused

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let z = i32(gid.x);
    let element = i32(gid.y);
    let H = params.x;
    let D = params.y;
    let filterSize = params.z;
    let fhalf = (filterSize - 1) / 2;
    if (z < fhalf || z >= D - fhalf || element >= 12) { return; }

    var sum: f32 = 0.0;
    let base = z * H + element * H * D;
    for (var y = fhalf; y < H - fhalf; y++) {
        sum += h2D[base + y];
    }
    h1D[z + element * D] = sum;
}
"""

# -- H-vector final reduction --

KERNEL_HVECTOR_FINAL = """
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read> h1D: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<i32>;  // D, filterSize, unused, unused

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = i32(gid.x);
    if (element >= 12) { return; }
    let D = params.x;
    let filterSize = params.y;
    let fhalf = (filterSize - 1) / 2;

    var sum: f32 = 0.0;
    let base = element * D;
    for (var z = fhalf; z < D - fhalf; z++) {
        sum += h1D[base + z];
    }
    h[element] = sum;
}
"""

# -- Tensor components (nonlinear) --

KERNEL_TENSOR_COMPONENTS = _HELPERS + """
struct TensorParams {
    m11: f32, m12: f32, m13: f32,
    m22: f32, m23: f32, m33: f32,
}

@group(0) @binding(0) var<storage, read_write> t11: array<f32>;
@group(0) @binding(1) var<storage, read_write> t12: array<f32>;
@group(0) @binding(2) var<storage, read_write> t13: array<f32>;
@group(0) @binding(3) var<storage, read_write> t22: array<f32>;
@group(0) @binding(4) var<storage, read_write> t23: array<f32>;
@group(0) @binding(5) var<storage, read_write> t33: array<f32>;
@group(0) @binding(6) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(7) var<uniform> dims: Dims;
@group(0) @binding(8) var<uniform> tp: TensorParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let q2v = q2[i];
    let mag = sqrt(q2v.x * q2v.x + q2v.y * q2v.y);

    t11[i] += mag * tp.m11;
    t12[i] += mag * tp.m12;
    t13[i] += mag * tp.m13;
    t22[i] += mag * tp.m22;
    t23[i] += mag * tp.m23;
    t33[i] += mag * tp.m33;
}
"""

# -- Tensor norms --

KERNEL_TENSOR_NORMS = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> norms: array<f32>;
@group(0) @binding(1) var<storage, read> t11: array<f32>;
@group(0) @binding(2) var<storage, read> t12: array<f32>;
@group(0) @binding(3) var<storage, read> t13: array<f32>;
@group(0) @binding(4) var<storage, read> t22: array<f32>;
@group(0) @binding(5) var<storage, read> t23: array<f32>;
@group(0) @binding(6) var<storage, read> t33: array<f32>;
@group(0) @binding(7) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }
    let i = idx3(x, y, z, dims.W, dims.H);
    let v11 = t11[i]; let v12 = t12[i]; let v13 = t13[i];
    let v22 = t22[i]; let v23 = t23[i]; let v33 = t33[i];
    norms[i] = sqrt(v11*v11 + 2.0*v12*v12 + 2.0*v13*v13 + v22*v22 + 2.0*v23*v23 + v33*v33);
}
"""

# -- Nonlinear A-matrices and h-vectors --

KERNEL_AMATRICES_HVECTORS = _HELPERS + """
struct MorphonParams {
    W: i32, H: i32, D: i32, FILTER: i32,
}

@group(0) @binding(0) var<storage, read_write> a11: array<f32>;
@group(0) @binding(1) var<storage, read_write> a12: array<f32>;
@group(0) @binding(2) var<storage, read_write> a13: array<f32>;
@group(0) @binding(3) var<storage, read_write> a22: array<f32>;
@group(0) @binding(4) var<storage, read_write> a23: array<f32>;
@group(0) @binding(5) var<storage, read_write> a33: array<f32>;
@group(0) @binding(6) var<storage, read_write> h1: array<f32>;
@group(0) @binding(7) var<storage, read_write> h2: array<f32>;
@group(0) @binding(8) var<storage, read_write> h3: array<f32>;
@group(0) @binding(9) var<storage, read> q1: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read> q2: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read> t11: array<f32>;
@group(0) @binding(12) var<storage, read> t12: array<f32>;
@group(0) @binding(13) var<storage, read> t13: array<f32>;
@group(0) @binding(14) var<storage, read> t22: array<f32>;
@group(0) @binding(15) var<storage, read> t23: array<f32>;
@group(0) @binding(16) var<storage, read> t33: array<f32>;
@group(0) @binding(17) var<storage, read> filterDirX: array<f32>;
@group(0) @binding(18) var<storage, read> filterDirY: array<f32>;
@group(0) @binding(19) var<storage, read> filterDirZ: array<f32>;
@group(0) @binding(20) var<uniform> p: MorphonParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= p.W || y >= p.H || z >= p.D) { return; }

    let i = idx3(x, y, z, p.W, p.H);
    let q1v = q1[i];
    let q2v = q2[i];

    let qqR = q1v.x * q2v.x + q1v.y * q2v.y;
    let qqI = -q1v.x * q2v.y + q1v.y * q2v.x;
    var pd = 0.0;
    if (abs(qqR) > 1e-30 || abs(qqI) > 1e-30) { pd = atan2(qqI, qqR); }
    let Aqq = qqR * qqR + qqI * qqI;
    let cosH = cos(pd * 0.5);
    var cert = 0.0;
    if (Aqq > 0.0) { cert = sqrt(sqrt(Aqq)) * cosH * cosH; }

    let T11 = t11[i]; let T12 = t12[i]; let T13 = t13[i];
    let T22 = t22[i]; let T23 = t23[i]; let T33 = t33[i];

    let tt11 = T11*T11 + T12*T12 + T13*T13;
    let tt12 = T11*T12 + T12*T22 + T13*T23;
    let tt13 = T11*T13 + T12*T23 + T13*T33;
    let tt22 = T12*T12 + T22*T22 + T23*T23;
    let tt23 = T12*T13 + T22*T23 + T23*T33;
    let tt33 = T13*T13 + T23*T23 + T33*T33;

    let fdx = filterDirX[p.FILTER];
    let fdy = filterDirY[p.FILTER];
    let fdz = filterDirZ[p.FILTER];

    let cpd = cert * pd;
    let hh1 = cpd * (fdx * tt11 + fdy * tt12 + fdz * tt13);
    let hh2 = cpd * (fdx * tt12 + fdy * tt22 + fdz * tt23);
    let hh3 = cpd * (fdx * tt13 + fdy * tt23 + fdz * tt33);

    if (p.FILTER == 0) {
        a11[i] = cert * tt11;
        a12[i] = cert * tt12;
        a13[i] = cert * tt13;
        a22[i] = cert * tt22;
        a23[i] = cert * tt23;
        a33[i] = cert * tt33;
        h1[i] = hh1;
        h2[i] = hh2;
        h3[i] = hh3;
    } else {
        a11[i] += cert * tt11;
        a12[i] += cert * tt12;
        a13[i] += cert * tt13;
        a22[i] += cert * tt22;
        a23[i] += cert * tt23;
        a33[i] += cert * tt33;
        h1[i] += hh1;
        h2[i] += hh2;
        h3[i] += hh3;
    }
}
"""

# -- Displacement update --

KERNEL_DISPLACEMENT_UPDATE = _HELPERS + """
@group(0) @binding(0) var<storage, read_write> dispX: array<f32>;
@group(0) @binding(1) var<storage, read_write> dispY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dispZ: array<f32>;
@group(0) @binding(3) var<storage, read> a11: array<f32>;
@group(0) @binding(4) var<storage, read> a12: array<f32>;
@group(0) @binding(5) var<storage, read> a13: array<f32>;
@group(0) @binding(6) var<storage, read> a22: array<f32>;
@group(0) @binding(7) var<storage, read> a23: array<f32>;
@group(0) @binding(8) var<storage, read> a33: array<f32>;
@group(0) @binding(9) var<storage, read> rh1: array<f32>;
@group(0) @binding(10) var<storage, read> rh2: array<f32>;
@group(0) @binding(11) var<storage, read> rh3: array<f32>;
@group(0) @binding(12) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let A11 = a11[i]; let A12 = a12[i]; let A13 = a13[i];
    let A22 = a22[i]; let A23 = a23[i]; let A33 = a33[i];
    let H1 = rh1[i]; let H2 = rh2[i]; let H3 = rh3[i];

    let det = A11*A22*A33 - A11*A23*A23 - A12*A12*A33
            + A12*A23*A13 + A13*A12*A23 - A13*A22*A13;

    let trace = A11 + A22 + A33;
    let epsilon = 0.01 * trace * trace * trace / 27.0 + 1e-16;
    let denom = det + epsilon;

    var dx = 0.0; var dy = 0.0; var dz = 0.0;
    if (abs(denom) > 1e-30) {
        let norm = 0.2 / denom;
        dx = norm * (H1*(A22*A33 - A23*A23) - H2*(A12*A33 - A13*A23) + H3*(A12*A23 - A13*A22));
        dy = norm * (H2*(A11*A33 - A13*A13) - H3*(A11*A23 - A13*A12) - H1*(A12*A33 - A23*A13));
        dz = norm * (H3*(A11*A22 - A12*A12) - H2*(A11*A23 - A12*A13) + H1*(A12*A23 - A22*A13));
        // Clamp extreme values
        if (abs(dx) > 1e6) { dx = 0.0; }
        if (abs(dy) > 1e6) { dy = 0.0; }
        if (abs(dz) > 1e6) { dz = 0.0; }
    }

    dispX[i] = dx;
    dispY[i] = dy;
    dispZ[i] = dz;
}
"""

# -- Interpolation: affine (linear registration) --

KERNEL_INTERPOLATE_LINEAR = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}
struct Dims { W: i32, H: i32, D: i32 }

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);
    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0+1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0+1, z0, W, H, D);
    let c110 = safe_read(x0+1, y0+1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0+1, W, H, D);
    let c101 = safe_read(x0+1, y0, z0+1, W, H, D);
    let c011 = safe_read(x0, y0+1, z0+1, W, H, D);
    let c111 = safe_read(x0+1, y0+1, z0+1, W, H, D);
    let c00 = c000*(1.0-fx) + c100*fx;
    let c10 = c010*(1.0-fx) + c110*fx;
    let c01 = c001*(1.0-fx) + c101*fx;
    let c11 = c011*(1.0-fx) + c111*fx;
    let c0 = c00*(1.0-fy) + c10*fy;
    let c1 = c01*(1.0-fy) + c11*fy;
    return c0*(1.0-fz) + c1*fz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let xf = f32(x) - (f32(dims.W) - 1.0) * 0.5;
    let yf = f32(y) - (f32(dims.H) - 1.0) * 0.5;
    let zf = f32(z) - (f32(dims.D) - 1.0) * 0.5;

    let px = f32(x) + params[0] + params[3]*xf + params[4]*yf + params[5]*zf;
    let py = f32(y) + params[1] + params[6]*xf + params[7]*yf + params[8]*zf;
    let pz = f32(z) + params[2] + params[9]*xf + params[10]*yf + params[11]*zf;

    output[idx3(x, y, z, dims.W, dims.H)] = trilinear(px, py, pz, dims.W, dims.H, dims.D);
}
"""

# -- Interpolation: nonlinear (displacement field) --

KERNEL_INTERPOLATE_NONLINEAR = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}
struct Dims { W: i32, H: i32, D: i32 }

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<storage, read> dX: array<f32>;
@group(0) @binding(3) var<storage, read> dY: array<f32>;
@group(0) @binding(4) var<storage, read> dZ: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);
    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0+1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0+1, z0, W, H, D);
    let c110 = safe_read(x0+1, y0+1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0+1, W, H, D);
    let c101 = safe_read(x0+1, y0, z0+1, W, H, D);
    let c011 = safe_read(x0, y0+1, z0+1, W, H, D);
    let c111 = safe_read(x0+1, y0+1, z0+1, W, H, D);
    let c00 = c000*(1.0-fx) + c100*fx;
    let c10 = c010*(1.0-fx) + c110*fx;
    let c01 = c001*(1.0-fx) + c101*fx;
    let c11 = c011*(1.0-fx) + c111*fx;
    let c0 = c00*(1.0-fy) + c10*fy;
    let c1 = c01*(1.0-fy) + c11*fy;
    return c0*(1.0-fz) + c1*fz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let i = idx3(x, y, z, dims.W, dims.H);
    let px = f32(x) + dX[i];
    let py = f32(y) + dY[i];
    let pz = f32(z) + dZ[i];

    output[i] = trilinear(px, py, pz, dims.W, dims.H, dims.D);
}
"""

# -- Rescale volume --

KERNEL_RESCALE_VOLUME = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct Dims { W: i32, H: i32, D: i32 }
struct Scales { x: f32, y: f32, z: f32 }

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;
@group(0) @binding(3) var<uniform> scales: Scales;
@group(0) @binding(4) var<uniform> srcDims: Dims;

fn safe_read(x: i32, y: i32, z: i32, W: i32, H: i32, D: i32) -> f32 {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0;
    }
    return volume[idx3(x, y, z, W, H)];
}

fn trilinear(px: f32, py: f32, pz: f32, W: i32, H: i32, D: i32) -> f32 {
    let x0 = i32(floor(px));
    let y0 = i32(floor(py));
    let z0 = i32(floor(pz));
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    let fz = pz - f32(z0);
    let c000 = safe_read(x0, y0, z0, W, H, D);
    let c100 = safe_read(x0+1, y0, z0, W, H, D);
    let c010 = safe_read(x0, y0+1, z0, W, H, D);
    let c110 = safe_read(x0+1, y0+1, z0, W, H, D);
    let c001 = safe_read(x0, y0, z0+1, W, H, D);
    let c101 = safe_read(x0+1, y0, z0+1, W, H, D);
    let c011 = safe_read(x0, y0+1, z0+1, W, H, D);
    let c111 = safe_read(x0+1, y0+1, z0+1, W, H, D);
    let c00 = c000*(1.0-fx) + c100*fx;
    let c10 = c010*(1.0-fx) + c110*fx;
    let c01 = c001*(1.0-fx) + c101*fx;
    let c11 = c011*(1.0-fx) + c111*fx;
    let c0 = c00*(1.0-fy) + c10*fy;
    let c1 = c01*(1.0-fy) + c11*fy;
    return c0*(1.0-fz) + c1*fz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let px = f32(x) * scales.x;
    let py = f32(y) * scales.y;
    let pz = f32(z) * scales.z;

    output[idx3(x, y, z, dims.W, dims.H)] = trilinear(px, py, pz, srcDims.W, srcDims.H, srcDims.D);
}
"""

# -- Copy volume to new dimensions --

KERNEL_COPY_VOLUME_TO_NEW = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}

struct CopyParams {
    newW: i32, newH: i32, newD: i32,
    srcW: i32, srcH: i32, srcD: i32,
    xDiff: i32, yDiff: i32, zDiff: i32,
    mmZCut: i32,
    voxelSizeZ: f32,
    _pad: i32,
}

@group(0) @binding(0) var<storage, read_write> newVol: array<f32>;
@group(0) @binding(1) var<storage, read> srcVol: array<f32>;
@group(0) @binding(2) var<uniform> p: CopyParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);

    var xNew: i32; var xSrc: i32;
    var yNew: i32; var ySrc: i32;
    var zNew: i32; var zSrc: i32;

    if (p.xDiff > 0) {
        xNew = x; xSrc = x + i32(round(f32(p.xDiff) / 2.0));
    } else {
        xNew = x + i32(round(f32(abs(p.xDiff)) / 2.0)); xSrc = x;
    }
    if (p.yDiff > 0) {
        yNew = y; ySrc = y + i32(round(f32(p.yDiff) / 2.0));
    } else {
        yNew = y + i32(round(f32(abs(p.yDiff)) / 2.0)); ySrc = y;
    }
    if (p.zDiff > 0) {
        zNew = z; zSrc = z + i32(round(f32(p.zDiff) / 2.0)) + i32(round(f32(p.mmZCut) / p.voxelSizeZ));
    } else {
        zNew = z + i32(round(f32(abs(p.zDiff)) / 2.0));
        zSrc = z + i32(round(f32(p.mmZCut) / p.voxelSizeZ));
    }

    if (xSrc < 0 || xSrc >= p.srcW || ySrc < 0 || ySrc >= p.srcH || zSrc < 0 || zSrc >= p.srcD) { return; }
    if (xNew < 0 || xNew >= p.newW || yNew < 0 || yNew >= p.newH || zNew < 0 || zNew >= p.newD) { return; }

    newVol[idx3(xNew, yNew, zNew, p.newW, p.newH)] = srcVol[idx3(xSrc, ySrc, zSrc, p.srcW, p.srcH)];
}
"""

# -- Add linear + nonlinear displacement --

KERNEL_ADD_LINEAR_NONLINEAR_DISP = """
fn idx3(x: i32, y: i32, z: i32, W: i32, H: i32) -> i32 {
    return x + y * W + z * W * H;
}
struct Dims { W: i32, H: i32, D: i32 }

@group(0) @binding(0) var<storage, read_write> dispX: array<f32>;
@group(0) @binding(1) var<storage, read_write> dispY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dispZ: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
    if (x >= dims.W || y >= dims.H || z >= dims.D) { return; }

    let xf = f32(x) - (f32(dims.W) - 1.0) * 0.5;
    let yf = f32(y) - (f32(dims.H) - 1.0) * 0.5;
    let zf = f32(z) - (f32(dims.D) - 1.0) * 0.5;

    let i = idx3(x, y, z, dims.W, dims.H);
    dispX[i] += params[0] + params[3]*xf + params[4]*yf + params[5]*zf;
    dispY[i] += params[1] + params[6]*xf + params[7]*yf + params[8]*zf;
    dispZ[i] += params[2] + params[9]*xf + params[10]*yf + params[11]*zf;
}
"""

# ============================================================
#  Kernel Registry
# ============================================================

KERNELS = {
    "fillFloat": KERNEL_FILL_FLOAT,
    "fillVec2": KERNEL_FILL_VEC2,
    "addVolumes": KERNEL_ADD_VOLUMES,
    "multiplyVolume": KERNEL_MULTIPLY_VOLUME,
    "multiplyVolumes": KERNEL_MULTIPLY_VOLUMES,
    "calculateColumnMaxs": KERNEL_COLUMN_MAXS,
    "calculateRowMaxs": KERNEL_ROW_MAXS,
    "conv3D_Full": KERNEL_CONV3D_FULL,
    "separableConvRows": KERNEL_SEPARABLE_CONV_ROWS,
    "separableConvColumns": KERNEL_SEPARABLE_CONV_COLS,
    "separableConvRods": KERNEL_SEPARABLE_CONV_RODS,
    "phaseDiffCert": KERNEL_PHASE_DIFF_CERT,
    "phaseGradX": KERNEL_PHASE_GRAD_X,
    "phaseGradY": KERNEL_PHASE_GRAD_Y,
    "phaseGradZ": KERNEL_PHASE_GRAD_Z,
    "amatrixHvector2D": KERNEL_AMATRIX_HVECTOR_2D,
    "amatrix1D": KERNEL_AMATRIX_1D,
    "amatrixFinal": KERNEL_AMATRIX_FINAL,
    "hvector1D": KERNEL_HVECTOR_1D,
    "hvectorFinal": KERNEL_HVECTOR_FINAL,
    "tensorComponents": KERNEL_TENSOR_COMPONENTS,
    "tensorNorms": KERNEL_TENSOR_NORMS,
    "amatricesHvectors": KERNEL_AMATRICES_HVECTORS,
    "displacementUpdate": KERNEL_DISPLACEMENT_UPDATE,
    "interpolateLinear": KERNEL_INTERPOLATE_LINEAR,
    "interpolateNonLinear": KERNEL_INTERPOLATE_NONLINEAR,
    "rescaleVolume": KERNEL_RESCALE_VOLUME,
    "copyVolumeToNew": KERNEL_COPY_VOLUME_TO_NEW,
    "addLinearNonLinearDisp": KERNEL_ADD_LINEAR_NONLINEAR_DISP,
}


# ============================================================
#  WebGPU Context
# ============================================================

class WebGPUContext:
    """Manages WebGPU device, queue, pipelines, and buffer operations."""

    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter = adapter
        # Request higher limits for large binding counts
        # wgpu-py uses hyphenated keys in adapter.limits
        al = adapter.limits
        self.device = adapter.request_device_sync(
            required_limits={
                "max-storage-buffers-per-shader-stage": min(21, al.get("max-storage-buffers-per-shader-stage", 8)),
                "max-buffer-size": al.get("max-buffer-size", 256 * 1024 * 1024),
                "max-storage-buffer-binding-size": al.get("max-storage-buffer-binding-size", 128 * 1024 * 1024),
            }
        )
        self._pipelines = {}
        self._shader_modules = {}

    def get_pipeline(self, name):
        if name not in self._pipelines:
            source = KERNELS[name]
            module = self.device.create_shader_module(code=source)
            pipeline = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": module, "entry_point": "main"},
            )
            self._pipelines[name] = pipeline
            self._shader_modules[name] = module
        return self._pipelines[name]

    def new_buffer(self, size_bytes, data=None):
        """Create a GPU buffer with STORAGE|COPY_SRC|COPY_DST usage."""
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        size_bytes = max(size_bytes, 4)  # Minimum buffer size
        buf = self.device.create_buffer(size=size_bytes, usage=usage)
        if data is not None:
            self.device.queue.write_buffer(buf, 0, data)
        return buf

    def new_uniform(self, data):
        """Create a uniform buffer from bytes."""
        usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        size = max(len(data), 4)
        buf = self.device.create_buffer(size=size, usage=usage)
        self.device.queue.write_buffer(buf, 0, data)
        return buf

    def read_buffer(self, buf):
        """Read buffer contents back to CPU as numpy float32 array."""
        data = self.device.queue.read_buffer(buf)
        return np.frombuffer(data, dtype=np.float32).copy()

    def write_buffer(self, buf, data):
        """Write numpy array to buffer."""
        self.device.queue.write_buffer(buf, 0, data.astype(np.float32).tobytes())

    def copy_buffer(self, src, dst, size_bytes):
        """GPU-side buffer copy."""
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size_bytes)
        self.device.queue.submit([encoder.finish()])

    def dispatch(self, kernel_name, bindings, workgroups):
        """Dispatch a single compute kernel."""
        pipeline = self.get_pipeline(kernel_name)
        entries = []
        for binding_idx, buf in bindings:
            entries.append({
                "binding": binding_idx,
                "resource": {"buffer": buf, "offset": 0, "size": buf.size},
            })
        bind_group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=entries,
        )
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)
        pass_enc.dispatch_workgroups(*workgroups)
        pass_enc.end()
        self.device.queue.submit([encoder.finish()])

    def multi_dispatch(self, dispatches):
        """Execute multiple dispatches in a single command encoder.

        Each dispatch is (kernel_name, bindings, workgroups).
        Storage writes from dispatch N are visible to dispatch N+1.
        """
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        for kernel_name, bindings, workgroups in dispatches:
            pipeline = self.get_pipeline(kernel_name)
            entries = []
            for binding_idx, buf in bindings:
                entries.append({
                    "binding": binding_idx,
                    "resource": {"buffer": buf, "offset": 0, "size": buf.size},
                })
            bind_group = self.device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=entries,
            )
            pass_enc.set_pipeline(pipeline)
            pass_enc.set_bind_group(0, bind_group)
            pass_enc.dispatch_workgroups(*workgroups)
        pass_enc.end()
        self.device.queue.submit([encoder.finish()])


# Singleton context
_ctx = None

def ctx():
    global _ctx
    if _ctx is None:
        _ctx = WebGPUContext()
    return _ctx


# ============================================================
#  Helper functions
# ============================================================

def _wg3d(W, H, D):
    return (math.ceil(W / 8), math.ceil(H / 8), D)

def _wg2d(W, H):
    return (math.ceil(W / 8), math.ceil(H / 8), 1)

def _wg1d(N):
    return (math.ceil(N / 256), 1, 1)

def _dims_uniform(c, W, H, D):
    return c.new_uniform(struct.pack("iii", W, H, D))


# ============================================================
#  GPU Operations
# ============================================================

def fill_buffer(c, buf, value, count):
    params = c.new_uniform(struct.pack("f", value))
    c.dispatch("fillFloat", [(0, buf), (1, params)], _wg1d(count))

def fill_vec2_buffer(c, buf, count):
    c.dispatch("fillVec2", [(0, buf)], _wg1d(count))

def add_volumes(c, A, B, count):
    c.dispatch("addVolumes", [(0, A), (1, B)], _wg1d(count))

def multiply_volume(c, vol, factor, count):
    params = c.new_uniform(struct.pack("f", factor))
    c.dispatch("multiplyVolume", [(0, vol), (1, params)], _wg1d(count))

def multiply_volumes(c, A, B, count):
    c.dispatch("multiplyVolumes", [(0, A), (1, B)], _wg1d(count))

def calculate_max(c, volume, W, H, D):
    dims = _dims_uniform(c, W, H, D)
    col_maxs = c.new_buffer(H * D * 4)
    row_maxs = c.new_buffer(D * 4)
    c.dispatch("calculateColumnMaxs",
               [(0, col_maxs), (1, volume), (2, dims)], _wg2d(H, D))
    c.dispatch("calculateRowMaxs",
               [(0, row_maxs), (1, col_maxs), (2, dims)], _wg1d(D))
    data = c.read_buffer(row_maxs)
    return float(np.max(data[:D]))


# ============================================================
#  Convolution
# ============================================================

def nonseparable_convolution_3d(c, resp1, resp2, resp3, volume,
                                 fReal1, fImag1, fReal2, fImag2, fReal3, fImag3,
                                 W, H, D):
    """3D 7x7x7 nonseparable convolution with 3 complex quadrature filters."""
    dims = _dims_uniform(c, W, H, D)
    f1r = c.new_buffer(343 * 4, fReal1.astype(np.float32))
    f1i = c.new_buffer(343 * 4, fImag1.astype(np.float32))
    f2r = c.new_buffer(343 * 4, fReal2.astype(np.float32))
    f2i = c.new_buffer(343 * 4, fImag2.astype(np.float32))
    f3r = c.new_buffer(343 * 4, fReal3.astype(np.float32))
    f3i = c.new_buffer(343 * 4, fImag3.astype(np.float32))

    c.dispatch("conv3D_Full", [
        (0, resp1), (1, resp2), (2, resp3), (3, volume),
        (4, f1r), (5, f1i), (6, f2r), (7, f2i), (8, f3r), (9, f3i),
        (10, dims),
    ], _wg3d(W, H, D))


# ============================================================
#  Smoothing
# ============================================================

def create_smoothing_filter(sigma):
    filt = np.zeros(9, dtype=np.float32)
    for i in range(9):
        x = float(i) - 4.0
        filt[i] = np.exp(-0.5 * x * x / (sigma * sigma))
    filt /= filt.sum()
    return filt

def perform_smoothing(c, output, input_buf, W, H, D, smooth_filter):
    dims = _dims_uniform(c, W, H, D)
    filt_buf = c.new_buffer(9 * 4, smooth_filter)
    vol = W * H * D
    temp1 = c.new_buffer(vol * 4)
    temp2 = c.new_buffer(vol * 4)

    # 3-pass separable: rows -> columns -> rods
    c.multi_dispatch([
        ("separableConvRows", [(0, temp1), (1, input_buf), (2, filt_buf), (3, dims)], _wg3d(W, H, D)),
        ("separableConvColumns", [(0, temp2), (1, temp1), (2, filt_buf), (3, dims)], _wg3d(W, H, D)),
        ("separableConvRods", [(0, output), (1, temp2), (2, filt_buf), (3, dims)], _wg3d(W, H, D)),
    ])

def perform_smoothing_in_place(c, volume, W, H, D, smooth_filter):
    vol = W * H * D
    output = c.new_buffer(vol * 4)
    perform_smoothing(c, output, volume, W, H, D, smooth_filter)
    c.copy_buffer(output, volume, vol * 4)

def batch_smooth_in_place(c, volumes, W, H, D, smooth_filter):
    for vol_buf in volumes:
        perform_smoothing_in_place(c, vol_buf, W, H, D, smooth_filter)


# ============================================================
#  Volume operations
# ============================================================

def rescale_volume(c, input_buf, srcW, srcH, srcD, dstW, dstH, dstD,
                   scaleX, scaleY, scaleZ):
    vol = dstW * dstH * dstD
    output = c.new_buffer(vol * 4)
    fill_buffer(c, output, 0.0, vol)
    dims = _dims_uniform(c, dstW, dstH, dstD)
    scales = c.new_uniform(struct.pack("fff", scaleX, scaleY, scaleZ))
    src_dims = c.new_uniform(struct.pack("iii", srcW, srcH, srcD))
    c.dispatch("rescaleVolume", [
        (0, output), (1, input_buf), (2, dims), (3, scales), (4, src_dims),
    ], _wg3d(dstW, dstH, dstD))
    return output

def copy_volume_to_new(c, src, srcW, srcH, srcD, dstW, dstH, dstD,
                        mmZCut, voxelSizeZ):
    vol = dstW * dstH * dstD
    dst = c.new_buffer(vol * 4)
    fill_buffer(c, dst, 0.0, vol)

    # Pack CopyParams: 10 ints + 1 float + 1 pad int = 48 bytes
    xDiff = srcW - dstW
    yDiff = srcH - dstH
    zDiff = srcD - dstD
    params = c.new_uniform(struct.pack("iiiiiiiiiifi",
        dstW, dstH, dstD, srcW, srcH, srcD,
        xDiff, yDiff, zDiff, mmZCut, voxelSizeZ, 0))

    dispW = max(srcW, dstW)
    dispH = max(srcH, dstH)
    dispD = max(srcD, dstD)

    c.dispatch("copyVolumeToNew", [(0, dst), (1, src), (2, params)],
               _wg3d(dispW, dispH, dispD))
    return dst

def change_volumes_resolution_and_size(c, input_buf, srcW, srcH, srcD, srcVox,
                                        dstW, dstH, dstD, dstVox, mmZCut):
    if mmZCut < 0: mmZCut = 0  # negative = disabled
    scaleX = srcVox[0] / dstVox[0]
    scaleY = srcVox[1] / dstVox[1]
    scaleZ = srcVox[2] / dstVox[2]

    interpW = int(round(srcW * scaleX))
    interpH = int(round(srcH * scaleY))
    interpD = int(round(srcD * scaleZ))

    voxDiffX = (srcW - 1) / max(interpW - 1, 1)
    voxDiffY = (srcH - 1) / max(interpH - 1, 1)
    voxDiffZ = (srcD - 1) / max(interpD - 1, 1)

    interpolated = rescale_volume(c, input_buf, srcW, srcH, srcD,
                                   interpW, interpH, interpD,
                                   voxDiffX, voxDiffY, voxDiffZ)

    return copy_volume_to_new(c, interpolated, interpW, interpH, interpD,
                               dstW, dstH, dstD, mmZCut, dstVox[2])

def change_volume_size(c, input_buf, srcW, srcH, srcD, dstW, dstH, dstD):
    scaleX = (srcW - 1) / max(dstW - 1, 1)
    scaleY = (srcH - 1) / max(dstH - 1, 1)
    scaleZ = (srcD - 1) / max(dstD - 1, 1)
    return rescale_volume(c, input_buf, srcW, srcH, srcD, dstW, dstH, dstD,
                          scaleX, scaleY, scaleZ)


# ============================================================
#  Interpolation
# ============================================================

def interpolate_linear(c, output, volume, params, W, H, D):
    """Affine interpolation with 12 parameters."""
    dims = _dims_uniform(c, W, H, D)
    params_buf = c.new_buffer(12 * 4, np.array(params, dtype=np.float32))
    # WebGPU doesn't allow same buffer as both read and read_write in one dispatch
    if output is volume:
        tmp = c.new_buffer(volume.size)
        c.copy_buffer(volume, tmp, volume.size)
        c.dispatch("interpolateLinear", [
            (0, output), (1, tmp), (2, params_buf), (3, dims),
        ], _wg3d(W, H, D))
    else:
        c.dispatch("interpolateLinear", [
            (0, output), (1, volume), (2, params_buf), (3, dims),
        ], _wg3d(W, H, D))

def interpolate_nonlinear(c, output, volume, dispX, dispY, dispZ, W, H, D):
    """Displacement field interpolation."""
    dims = _dims_uniform(c, W, H, D)
    if output is volume:
        tmp = c.new_buffer(volume.size)
        c.copy_buffer(volume, tmp, volume.size)
        c.dispatch("interpolateNonLinear", [
            (0, output), (1, tmp), (2, dispX), (3, dispY), (4, dispZ), (5, dims),
        ], _wg3d(W, H, D))
    else:
        c.dispatch("interpolateNonLinear", [
            (0, output), (1, volume), (2, dispX), (3, dispY), (4, dispZ), (5, dims),
        ], _wg3d(W, H, D))

def add_linear_nonlinear_displacement(c, dispX, dispY, dispZ, params, W, H, D):
    dims = _dims_uniform(c, W, H, D)
    params_buf = c.new_buffer(12 * 4, np.array(params, dtype=np.float32))
    c.dispatch("addLinearNonLinearDisp", [
        (0, dispX), (1, dispY), (2, dispZ), (3, params_buf), (4, dims),
    ], _wg3d(W, H, D))


# ============================================================
#  CPU helpers
# ============================================================

def center_of_mass(vol_data, W, H, D):
    """Compute center of mass (CPU-side)."""
    total = 0.0
    sx = sy = sz = 0.0
    for z in range(D):
        for y in range(H):
            for x in range(W):
                v = vol_data[x + y * W + z * W * H]
                if v > 0:
                    total += v
                    sx += v * x
                    sy += v * y
                    sz += v * z
    if total > 0:
        return sx / total, sy / total, sz / total
    return W * 0.5, H * 0.5, D * 0.5

def solve_equation_system(A_flat, h_vec, n=12):
    """Solve Ax = h via Gaussian elimination with partial pivoting (double precision)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = A_flat[j * n + i]  # column-major to row-major
    # Mirror symmetric
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i]

    h = np.array(h_vec[:n], dtype=np.float64)

    # Augmented matrix
    aug = np.zeros((n, n + 1), dtype=np.float64)
    aug[:, :n] = A
    aug[:, n] = h

    # Forward elimination with partial pivoting
    for col in range(n):
        pivot_row = col
        pivot_val = abs(aug[col, col])
        for row in range(col + 1, n):
            if abs(aug[row, col]) > pivot_val:
                pivot_val = abs(aug[row, col])
                pivot_row = row
        if pivot_val < 1e-30:
            return np.zeros(n, dtype=np.float64)
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            aug[row, col:] -= factor * aug[col, col:]

    # Back substitution
    params = np.zeros(n, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        s = aug[row, n]
        for j in range(row + 1, n):
            s -= aug[row, j] * params[j]
        params[row] = s / aug[row, row]
    return params


# ============================================================
#  Affine parameter composition
# ============================================================

def params_to_matrix(p, translation_scale=1.0):
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = p[3] + 1.0; M[0, 1] = p[4];       M[0, 2] = p[5];       M[0, 3] = p[0] * translation_scale
    M[1, 0] = p[6];        M[1, 1] = p[7] + 1.0; M[1, 2] = p[8];       M[1, 3] = p[1] * translation_scale
    M[2, 0] = p[9];        M[2, 1] = p[10];      M[2, 2] = p[11] + 1.0; M[2, 3] = p[2] * translation_scale
    return M

def matrix_to_params(M):
    p = np.zeros(12, dtype=np.float32)
    p[0] = M[0, 3]; p[1] = M[1, 3]; p[2] = M[2, 3]
    p[3] = M[0, 0] - 1.0; p[4] = M[0, 1]; p[5] = M[0, 2]
    p[6] = M[1, 0]; p[7] = M[1, 1] - 1.0; p[8] = M[1, 2]
    p[9] = M[2, 0]; p[10] = M[2, 1]; p[11] = M[2, 2] - 1.0
    return p

def compose_affine_params(old_params, new_params, translation_scale=1.0):
    O = params_to_matrix(old_params, translation_scale)
    N = params_to_matrix(new_params, translation_scale)
    T = N @ O
    return matrix_to_params(T)


# ============================================================
#  Linear Registration
# ============================================================

def align_two_volumes_linear(c, aligned_buf, ref_buf, filters,
                              W, H, D, filter_size, num_iterations, verbose=False):
    """Single-scale linear registration."""
    vol = W * H * D
    reg_params = np.zeros(12, dtype=np.float32)

    # Save original aligned volume
    original = c.new_buffer(vol * 4)
    c.copy_buffer(aligned_buf, original, vol * 4)

    # Allocate filter response buffers (complex = vec2)
    q11 = c.new_buffer(vol * 8); q12 = c.new_buffer(vol * 8); q13 = c.new_buffer(vol * 8)
    q21 = c.new_buffer(vol * 8); q22 = c.new_buffer(vol * 8); q23 = c.new_buffer(vol * 8)

    # Phase/certainty buffers
    phase_diff = c.new_buffer(vol * 4)
    certainties = c.new_buffer(vol * 4)
    phase_grad = c.new_buffer(vol * 4)

    # A-matrix / h-vector buffers
    HD = H * D
    A2D = c.new_buffer(30 * HD * 4)
    A1D = c.new_buffer(30 * D * 4)
    Amat = c.new_buffer(144 * 4)
    h2D = c.new_buffer(12 * HD * 4)
    h1D = c.new_buffer(12 * D * 4)
    hvec = c.new_buffer(12 * 4)

    # Filter reference volume once
    nonseparable_convolution_3d(c, q11, q12, q13, ref_buf,
        filters['linearReal'][0], filters['linearImag'][0],
        filters['linearReal'][1], filters['linearImag'][1],
        filters['linearReal'][2], filters['linearImag'][2],
        W, H, D)

    dims = _dims_uniform(c, W, H, D)
    grad_kernels = ["phaseGradX", "phaseGradY", "phaseGradZ"]
    dir_offsets = [0, 10, 20]
    h_offsets = [0, 1, 2]
    q1_bufs = [q11, q12, q13]
    q2_bufs = [q21, q22, q23]

    for iteration in range(num_iterations):
        # Filter aligned volume
        nonseparable_convolution_3d(c, q21, q22, q23, aligned_buf,
            filters['linearReal'][0], filters['linearImag'][0],
            filters['linearReal'][1], filters['linearImag'][1],
            filters['linearReal'][2], filters['linearImag'][2],
            W, H, D)

        # Zero intermediate buffers
        fill_buffer(c, A2D, 0.0, 30 * HD)
        fill_buffer(c, h2D, 0.0, 12 * HD)

        # Process each direction
        for d in range(3):
            # Phase differences + certainties
            c.dispatch("phaseDiffCert", [
                (0, phase_diff), (1, certainties), (2, q1_bufs[d]), (3, q2_bufs[d]), (4, dims),
            ], _wg3d(W, H, D))

            # Phase gradients
            c.dispatch(grad_kernels[d], [
                (0, phase_grad), (1, q1_bufs[d]), (2, q2_bufs[d]), (3, dims),
            ], _wg3d(W, H, D))

            # A-matrix and h-vector 2D
            ap = c.new_uniform(struct.pack("iiiiii",
                W, H, D, filter_size, dir_offsets[d], h_offsets[d]))
            c.dispatch("amatrixHvector2D", [
                (0, A2D), (1, h2D), (2, phase_diff), (3, phase_grad),
                (4, certainties), (5, ap),
            ], _wg2d(H, D))

        # Reduce A-matrix: 2D -> 1D -> Final
        hd_params = c.new_uniform(struct.pack("iiii", H, D, filter_size, 0))
        c.dispatch("amatrix1D", [(0, A1D), (1, A2D), (2, hd_params)], _wg2d(D, 30))
        c.dispatch("hvector1D", [(0, h1D), (1, h2D), (2, hd_params)], _wg2d(D, 12))

        fill_buffer(c, Amat, 0.0, 144)
        df_params = c.new_uniform(struct.pack("iiii", D, filter_size, 0, 0))
        c.dispatch("amatrixFinal", [(0, Amat), (1, A1D), (2, df_params)], (1, 1, 1))

        # Read back A and h, solve on CPU
        A_data = c.read_buffer(Amat)[:144]

        # h-vector final reduction on CPU (matches Metal)
        h1D_data = c.read_buffer(h1D)
        fhalf = (filter_size - 1) // 2
        h_data = np.zeros(12, dtype=np.float32)
        for elem in range(12):
            s = 0.0
            for z in range(fhalf, D - fhalf):
                s += h1D_data[elem * D + z]
            h_data[elem] = s

        params_dbl = solve_equation_system(A_data, h_data, 12)

        delta_params = params_dbl.astype(np.float32)
        reg_params = compose_affine_params(reg_params, delta_params)

        # Apply affine transform from original volume
        interpolate_linear(c, aligned_buf, original, reg_params, W, H, D)

    return reg_params


def align_two_volumes_linear_several_scales(c, aligned_buf, ref_buf, filters,
                                             W, H, D, filter_size, num_iterations,
                                             coarsest_scale, verbose=False):
    """Multi-scale linear registration."""
    vol = W * H * D
    reg_params = np.zeros(12, dtype=np.float32)

    # Keep original full-resolution aligned volume
    original = c.new_buffer(vol * 4)
    c.copy_buffer(aligned_buf, original, vol * 4)

    scale = coarsest_scale
    while scale >= 1:
        sW = int(round(W / scale))
        sH = int(round(H / scale))
        sD = int(round(D / scale))

        if sW < 8 or sH < 8 or sD < 8:
            scale //= 2
            continue

        if verbose:
            print(f"  Linear scale {scale}: {sW}x{sH}x{sD}")

        # Downscale from originals
        if scale == 1:
            scaled_ref = ref_buf
            scaled_aligned = aligned_buf
        else:
            scaled_ref = change_volume_size(c, ref_buf, W, H, D, sW, sH, sD)
            scaled_aligned = change_volume_size(c, original, W, H, D, sW, sH, sD)

        # Pre-transform with accumulated params (non-coarsest)
        if scale < coarsest_scale:
            interpolate_linear(c, scaled_aligned, scaled_aligned, reg_params, sW, sH, sD)

        iters = num_iterations if scale != 1 else int(math.ceil(num_iterations / 5.0))

        temp_params = align_two_volumes_linear(c, scaled_aligned, scaled_ref, filters,
                                                sW, sH, sD, filter_size, iters, verbose)

        # Compose
        if scale != 1:
            reg_params = compose_affine_params(reg_params, temp_params, translation_scale=2.0)
        else:
            reg_params = compose_affine_params(reg_params, temp_params)

        scale //= 2

    # Final transform at full resolution
    interpolate_linear(c, aligned_buf, original, reg_params, W, H, D)
    return reg_params


# ============================================================
#  Nonlinear Registration
# ============================================================

def align_two_volumes_nonlinear(c, aligned_buf, ref_buf, filters,
                                 W, H, D, num_iterations,
                                 update_disp_x, update_disp_y, update_disp_z,
                                 verbose=False):
    """Single-scale nonlinear registration."""
    vol = W * H * D

    # Allocate buffers
    q1 = [c.new_buffer(vol * 8) for _ in range(6)]
    q2 = [c.new_buffer(vol * 8) for _ in range(6)]
    t11 = c.new_buffer(vol * 4); t12 = c.new_buffer(vol * 4)
    t13 = c.new_buffer(vol * 4); t22 = c.new_buffer(vol * 4)
    t23 = c.new_buffer(vol * 4); t33 = c.new_buffer(vol * 4)
    a11 = c.new_buffer(vol * 4); a12 = c.new_buffer(vol * 4)
    a13 = c.new_buffer(vol * 4); a22 = c.new_buffer(vol * 4)
    a23 = c.new_buffer(vol * 4); a33 = c.new_buffer(vol * 4)
    h1 = c.new_buffer(vol * 4); h2 = c.new_buffer(vol * 4); h3 = c.new_buffer(vol * 4)
    tensor_norms = c.new_buffer(vol * 4)
    dux = c.new_buffer(vol * 4); duy = c.new_buffer(vol * 4); duz = c.new_buffer(vol * 4)

    original = c.new_buffer(vol * 4)
    c.copy_buffer(aligned_buf, original, vol * 4)

    # Filter direction buffers
    fdx_buf = c.new_buffer(6 * 4, np.array(filters['filterDirectionsX'], dtype=np.float32))
    fdy_buf = c.new_buffer(6 * 4, np.array(filters['filterDirectionsY'], dtype=np.float32))
    fdz_buf = c.new_buffer(6 * 4, np.array(filters['filterDirectionsZ'], dtype=np.float32))

    smooth_tensor = create_smoothing_filter(1.0)
    smooth_eq = create_smoothing_filter(2.0)
    smooth_disp = create_smoothing_filter(2.0)

    # Filter reference volume (once, 2 batches of 3 filters)
    nonseparable_convolution_3d(c, q1[0], q1[1], q1[2], ref_buf,
        filters['nonlinearReal'][0], filters['nonlinearImag'][0],
        filters['nonlinearReal'][1], filters['nonlinearImag'][1],
        filters['nonlinearReal'][2], filters['nonlinearImag'][2],
        W, H, D)
    nonseparable_convolution_3d(c, q1[3], q1[4], q1[5], ref_buf,
        filters['nonlinearReal'][3], filters['nonlinearImag'][3],
        filters['nonlinearReal'][4], filters['nonlinearImag'][4],
        filters['nonlinearReal'][5], filters['nonlinearImag'][5],
        W, H, D)

    for iteration in range(num_iterations):
        if verbose:
            print(f"    Nonlinear iter {iteration + 1}/{num_iterations}")

        # Filter aligned volume
        nonseparable_convolution_3d(c, q2[0], q2[1], q2[2], aligned_buf,
            filters['nonlinearReal'][0], filters['nonlinearImag'][0],
            filters['nonlinearReal'][1], filters['nonlinearImag'][1],
            filters['nonlinearReal'][2], filters['nonlinearImag'][2],
            W, H, D)
        nonseparable_convolution_3d(c, q2[3], q2[4], q2[5], aligned_buf,
            filters['nonlinearReal'][3], filters['nonlinearImag'][3],
            filters['nonlinearReal'][4], filters['nonlinearImag'][4],
            filters['nonlinearReal'][5], filters['nonlinearImag'][5],
            W, H, D)

        # Zero tensors and displacement update
        for buf in [t11, t12, t13, t22, t23, t33, dux, duy, duz]:
            fill_buffer(c, buf, 0.0, vol)

        # Compute tensor components (6 filters)
        dims = _dims_uniform(c, W, H, D)
        for f in range(6):
            pt = filters['projectionTensors'][f]
            tp = c.new_uniform(struct.pack("ffffff", *pt))
            c.dispatch("tensorComponents", [
                (0, t11), (1, t12), (2, t13), (3, t22), (4, t23), (5, t33),
                (6, q2[f]), (7, dims), (8, tp),
            ], _wg3d(W, H, D))

        # Tensor norms (pre-smooth)
        c.dispatch("tensorNorms", [
            (0, tensor_norms), (1, t11), (2, t12), (3, t13),
            (4, t22), (5, t23), (6, t33), (7, dims),
        ], _wg3d(W, H, D))

        # Smooth tensors
        batch_smooth_in_place(c, [t11, t12, t13, t22, t23, t33], W, H, D, smooth_tensor)

        # Tensor norms (post-smooth) + normalize
        dims = _dims_uniform(c, W, H, D)
        c.dispatch("tensorNorms", [
            (0, tensor_norms), (1, t11), (2, t12), (3, t13),
            (4, t22), (5, t23), (6, t33), (7, dims),
        ], _wg3d(W, H, D))

        max_norm = calculate_max(c, tensor_norms, W, H, D)
        if max_norm > 0:
            inv_max = 1.0 / max_norm
            for buf in [t11, t12, t13, t22, t23, t33]:
                multiply_volume(c, buf, inv_max, vol)

        # A-matrices and h-vectors (6 filters)
        for f in range(6):
            mp = c.new_uniform(struct.pack("iiii", W, H, D, f))
            c.dispatch("amatricesHvectors", [
                (0, a11), (1, a12), (2, a13), (3, a22), (4, a23), (5, a33),
                (6, h1), (7, h2), (8, h3),
                (9, q1[f]), (10, q2[f]),
                (11, t11), (12, t12), (13, t13), (14, t22), (15, t23), (16, t33),
                (17, fdx_buf), (18, fdy_buf), (19, fdz_buf),
                (20, mp),
            ], _wg3d(W, H, D))

        # Smooth A-matrix and h-vector
        batch_smooth_in_place(c, [a11, a12, a13, a22, a23, a33, h1, h2, h3], W, H, D, smooth_eq)

        # Displacement update
        dims = _dims_uniform(c, W, H, D)
        c.dispatch("displacementUpdate", [
            (0, dux), (1, duy), (2, duz),
            (3, a11), (4, a12), (5, a13), (6, a22), (7, a23), (8, a33),
            (9, h1), (10, h2), (11, h3), (12, dims),
        ], _wg3d(W, H, D))

        # Smooth displacement
        batch_smooth_in_place(c, [dux, duy, duz], W, H, D, smooth_disp)

        # Accumulate displacement
        add_volumes(c, update_disp_x, dux, vol)
        add_volumes(c, update_disp_y, duy, vol)
        add_volumes(c, update_disp_z, duz, vol)

        # Interpolate from original with accumulated displacement
        interpolate_nonlinear(c, aligned_buf, original,
                              update_disp_x, update_disp_y, update_disp_z, W, H, D)


def align_two_volumes_nonlinear_several_scales(c, aligned_buf, ref_buf, filters,
                                                W, H, D, num_iterations,
                                                coarsest_scale, verbose=False):
    """Multi-scale nonlinear registration."""
    vol = W * H * D

    original = c.new_buffer(vol * 4)
    c.copy_buffer(aligned_buf, original, vol * 4)

    total_disp_x = c.new_buffer(vol * 4)
    total_disp_y = c.new_buffer(vol * 4)
    total_disp_z = c.new_buffer(vol * 4)
    fill_buffer(c, total_disp_x, 0.0, vol)
    fill_buffer(c, total_disp_y, 0.0, vol)
    fill_buffer(c, total_disp_z, 0.0, vol)

    scale = coarsest_scale
    while scale >= 1:
        sW = W // scale
        sH = H // scale
        sD = D // scale

        if sW < 8 or sH < 8 or sD < 8:
            scale //= 2
            continue

        if verbose:
            print(f"  Nonlinear scale {scale}: {sW}x{sH}x{sD}")

        if scale == 1:
            scaled_ref = ref_buf
            scaled_aligned = aligned_buf
        else:
            scaled_ref = change_volume_size(c, ref_buf, W, H, D, sW, sH, sD)
            scaled_aligned = change_volume_size(c, aligned_buf, W, H, D, sW, sH, sD)

        sVol = sW * sH * sD
        update_x = c.new_buffer(sVol * 4)
        update_y = c.new_buffer(sVol * 4)
        update_z = c.new_buffer(sVol * 4)
        fill_buffer(c, update_x, 0.0, sVol)
        fill_buffer(c, update_y, 0.0, sVol)
        fill_buffer(c, update_z, 0.0, sVol)

        align_two_volumes_nonlinear(c, scaled_aligned, scaled_ref, filters,
                                     sW, sH, sD, num_iterations,
                                     update_x, update_y, update_z, verbose)

        if scale > 1:
            resc_x = change_volume_size(c, update_x, sW, sH, sD, W, H, D)
            resc_y = change_volume_size(c, update_y, sW, sH, sD, W, H, D)
            resc_z = change_volume_size(c, update_z, sW, sH, sD, W, H, D)
            multiply_volume(c, resc_x, float(scale), vol)
            multiply_volume(c, resc_y, float(scale), vol)
            multiply_volume(c, resc_z, float(scale), vol)
            add_volumes(c, total_disp_x, resc_x, vol)
            add_volumes(c, total_disp_y, resc_y, vol)
            add_volumes(c, total_disp_z, resc_z, vol)

            interpolate_nonlinear(c, aligned_buf, original,
                                  total_disp_x, total_disp_y, total_disp_z, W, H, D)
        else:
            add_volumes(c, total_disp_x, update_x, vol)
            add_volumes(c, total_disp_y, update_y, vol)
            add_volumes(c, total_disp_z, update_z, vol)

        scale //= 2

    return total_disp_x, total_disp_y, total_disp_z


# ============================================================
#  Public API
# ============================================================

def registerEPIT1(epi_data, epi_voxel_size, t1_data, t1_voxel_size,
                  parametric_filters, nonparametric_filters,
                  projection_tensors, filter_directions,
                  num_iterations=20, coarsest_scale=8, mm_z_cut=0,
                  opencl_platform=0, opencl_device=0, verbose=False):
    """Register EPI to T1 using parametric (phase-based) registration.

    API matches metal_registration.registerEPIT1().
    """
    c = ctx()

    # Extract dimensions (packed format: D, H, W)
    epiD, epiH, epiW = epi_data.shape
    t1D, t1H, t1W = t1_data.shape

    epi_vox = epi_voxel_size.astype(np.float32)
    t1_vox = t1_voxel_size.astype(np.float32)

    # Build filters dict
    filters = _build_filters_dict(parametric_filters, nonparametric_filters,
                                   projection_tensors, filter_directions)

    t1_vol = t1W * t1H * t1D

    # Upload volumes
    t1_buf = c.new_buffer(t1_vol * 4, t1_data.astype(np.float32))
    epi_buf = c.new_buffer(epiW * epiH * epiD * 4, epi_data.astype(np.float32))

    # Resample EPI to T1 resolution
    epi_in_t1 = change_volumes_resolution_and_size(
        c, epi_buf, epiW, epiH, epiD, epi_vox,
        t1W, t1H, t1D, t1_vox, mm_z_cut)

    # Center-of-mass alignment
    t1_cpu = c.read_buffer(t1_buf)[:t1_vol]
    epi_cpu = c.read_buffer(epi_in_t1)[:t1_vol]
    cx1, cy1, cz1 = center_of_mass(t1_cpu, t1W, t1H, t1D)
    cx2, cy2, cz2 = center_of_mass(epi_cpu, t1W, t1H, t1D)

    init_params = np.zeros(12, dtype=np.float32)
    # Round to integers to avoid interpolation blur (matches OpenCL's myround)
    init_params[0] = round(cx2 - cx1)
    init_params[1] = round(cy2 - cy1)
    init_params[2] = round(cz2 - cz1)
    interpolate_linear(c, epi_in_t1, epi_in_t1, init_params, t1W, t1H, t1D)

    # Save interpolated volume
    interp_result = c.read_buffer(epi_in_t1)[:t1_vol]

    # Multi-scale linear registration
    reg_params = align_two_volumes_linear_several_scales(
        c, epi_in_t1, t1_buf, filters,
        t1W, t1H, t1D, 7, num_iterations, coarsest_scale, verbose)

    # Read results
    aligned = c.read_buffer(epi_in_t1)[:t1_vol].reshape(t1D, t1H, t1W)
    interpolated = interp_result.reshape(t1D, t1H, t1W)
    params = reg_params[:6].copy()

    return aligned, interpolated, params


def registerT1MNI(t1_data, t1_voxel_size, mni_data, mni_voxel_size,
                  mni_brain_data, mni_mask_data,
                  parametric_filters, nonparametric_filters,
                  projection_tensors, filter_directions,
                  linear_iterations=10, nonlinear_iterations=5,
                  coarsest_scale=4, mm_z_cut=0,
                  opencl_platform=0, opencl_device=0, verbose=False):
    """Register T1 to MNI using affine + nonlinear registration.

    API matches metal_registration.registerT1MNI().
    """
    c = ctx()

    t1D, t1H, t1W = t1_data.shape
    mniD, mniH, mniW = mni_data.shape

    t1_vox = t1_voxel_size.astype(np.float32)
    mni_vox = mni_voxel_size.astype(np.float32)

    filters = _build_filters_dict(parametric_filters, nonparametric_filters,
                                   projection_tensors, filter_directions)

    mni_vol = mniW * mniH * mniD

    # Upload volumes
    mni_buf = c.new_buffer(mni_vol * 4, mni_data.astype(np.float32))
    mni_brain_buf = c.new_buffer(mni_vol * 4, mni_brain_data.astype(np.float32))
    mni_mask_buf = c.new_buffer(mni_vol * 4, mni_mask_data.astype(np.float32))
    t1_buf = c.new_buffer(t1W * t1H * t1D * 4, t1_data.astype(np.float32))

    # Resample T1 to MNI resolution
    t1_in_mni = change_volumes_resolution_and_size(
        c, t1_buf, t1W, t1H, t1D, t1_vox,
        mniW, mniH, mniD, mni_vox, mm_z_cut)

    # Center-of-mass alignment
    mni_cpu = c.read_buffer(mni_buf)[:mni_vol]
    t1_cpu = c.read_buffer(t1_in_mni)[:mni_vol]
    cx1, cy1, cz1 = center_of_mass(mni_cpu, mniW, mniH, mniD)
    cx2, cy2, cz2 = center_of_mass(t1_cpu, mniW, mniH, mniD)

    init_params = np.zeros(12, dtype=np.float32)
    # Round to integers to avoid interpolation blur (matches OpenCL's myround)
    init_params[0] = round(cx2 - cx1)
    init_params[1] = round(cy2 - cy1)
    init_params[2] = round(cz2 - cz1)
    interpolate_linear(c, t1_in_mni, t1_in_mni, init_params, mniW, mniH, mniD)

    # Save interpolated volume
    interp_result = c.read_buffer(t1_in_mni)[:mni_vol]

    # Linear registration
    reg_params = align_two_volumes_linear_several_scales(
        c, t1_in_mni, mni_buf, filters,
        mniW, mniH, mniD, 7, linear_iterations, coarsest_scale, verbose)

    # Compose COM shift into registration params (matches OpenCL line 8042)
    reg_params = compose_affine_params(reg_params, init_params)

    # Save linear result — re-rescale original T1 and apply combined
    # COM + affine in a single interpolation to avoid compounded blur
    fresh_t1 = change_volumes_resolution_and_size(
        c, t1_buf, t1W, t1H, t1D, t1_vox,
        mniW, mniH, mniD, mni_vox, mm_z_cut)
    interpolate_linear(c, fresh_t1, fresh_t1, reg_params, mniW, mniH, mniD)
    aligned_linear = c.read_buffer(fresh_t1)[:mni_vol].reshape(mniD, mniH, mniW)

    # Nonlinear registration
    if nonlinear_iterations > 0:
        total_dx, total_dy, total_dz = align_two_volumes_nonlinear_several_scales(
            c, t1_in_mni, mni_buf, filters,
            mniW, mniH, mniD, nonlinear_iterations, coarsest_scale, verbose)

        # Combine linear + nonlinear displacement
        add_linear_nonlinear_displacement(c, total_dx, total_dy, total_dz,
                                           reg_params, mniW, mniH, mniD)

        disp_x = c.read_buffer(total_dx)[:mni_vol].reshape(mniD, mniH, mniW)
        disp_y = c.read_buffer(total_dy)[:mni_vol].reshape(mniD, mniH, mniW)
        disp_z = c.read_buffer(total_dz)[:mni_vol].reshape(mniD, mniH, mniW)

        # Re-rescale original T1 and apply combined displacement in one step
        fresh_t1_nl = change_volumes_resolution_and_size(
            c, t1_buf, t1W, t1H, t1D, t1_vox,
            mniW, mniH, mniD, mni_vox, mm_z_cut)
        interpolate_nonlinear(c, fresh_t1_nl, fresh_t1_nl,
                              total_dx, total_dy, total_dz, mniW, mniH, mniD)
        aligned_nonlinear = c.read_buffer(fresh_t1_nl)[:mni_vol].reshape(mniD, mniH, mniW)

        # Skullstrip from the single-step result
        multiply_volumes(c, fresh_t1_nl, mni_mask_buf, mni_vol)
        skullstripped = c.read_buffer(fresh_t1_nl)[:mni_vol].reshape(mniD, mniH, mniW)
    else:
        aligned_nonlinear = aligned_linear.copy()
        skullstripped = np.zeros((mniD, mniH, mniW), dtype=np.float32)
        disp_x = np.zeros((mniD, mniH, mniW), dtype=np.float32)
        disp_y = np.zeros((mniD, mniH, mniW), dtype=np.float32)
        disp_z = np.zeros((mniD, mniH, mniW), dtype=np.float32)

    interpolated = interp_result.reshape(mniD, mniH, mniW)
    params_out = reg_params[:12].copy()

    return (aligned_linear, aligned_nonlinear, skullstripped, interpolated,
            params_out, disp_x, disp_y, disp_z)


# ============================================================
#  Filter helpers
# ============================================================

def _build_filters_dict(parametric_filters, nonparametric_filters,
                         projection_tensors, filter_directions):
    """Convert filter lists to internal dict format."""
    filters = {
        'linearReal': [],
        'linearImag': [],
        'nonlinearReal': [],
        'nonlinearImag': [],
        'projectionTensors': [],
        'filterDirectionsX': np.zeros(6, dtype=np.float32),
        'filterDirectionsY': np.zeros(6, dtype=np.float32),
        'filterDirectionsZ': np.zeros(6, dtype=np.float32),
    }

    # Parametric filters (3, complex interleaved: shape (343, 2))
    for i in range(3):
        f = np.asarray(parametric_filters[i], dtype=np.float32)
        if f.ndim == 2 and f.shape[1] == 2:
            filters['linearReal'].append(f[:, 0].copy())
            filters['linearImag'].append(f[:, 1].copy())
        elif f.size == 343 * 2:
            f = f.reshape(-1, 2)
            filters['linearReal'].append(f[:, 0].copy())
            filters['linearImag'].append(f[:, 1].copy())
        else:
            filters['linearReal'].append(f.flatten()[:343].copy())
            filters['linearImag'].append(np.zeros(343, dtype=np.float32))

    # Nonparametric filters (6)
    for i in range(6):
        f = np.asarray(nonparametric_filters[i], dtype=np.float32)
        if f.size >= 343 * 2:
            f = f.reshape(-1, 2)
            filters['nonlinearReal'].append(f[:, 0].copy())
            filters['nonlinearImag'].append(f[:, 1].copy())
        else:
            filters['nonlinearReal'].append(f.flatten()[:343].copy())
            filters['nonlinearImag'].append(np.zeros(343, dtype=np.float32))

    # Projection tensors (6 x 6: m11,m12,m13,m22,m23,m33)
    for i in range(6):
        pt = np.asarray(projection_tensors[i], dtype=np.float32)
        filters['projectionTensors'].append(pt[:6].copy())

    # Filter directions (3 arrays of 6)
    for d, key in enumerate(['filterDirectionsX', 'filterDirectionsY', 'filterDirectionsZ']):
        fd = np.asarray(filter_directions[d], dtype=np.float32)
        filters[key] = fd[:6].copy()

    return filters
