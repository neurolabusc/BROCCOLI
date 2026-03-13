// pybind11 module for Metal registration
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../src/metal_registration.h"

namespace py = pybind11;
using namespace metal_reg;

// Helper to convert numpy array to raw float pointer
static const float* getFloatPtr(py::array_t<float>& arr) {
    auto buf = arr.request();
    return static_cast<const float*>(buf.ptr);
}

static py::tuple py_registerEPIT1(
    py::array_t<float> epiData, py::array_t<float> epiVoxelSize,
    py::array_t<float> t1Data, py::array_t<float> t1VoxelSize,
    py::list parametricFilters,
    py::list nonparametricFilters,
    py::list projectionTensors,
    py::list filterDirections,
    int numIterations, int coarsestScale, int mmZCut,
    int opencl_platform, int opencl_device, bool verbose)
{
    // Extract volume dimensions from numpy array shapes
    auto epiBuf = epiData.request();
    auto t1Buf = t1Data.request();

    VolumeDims epiDims, t1Dims;
    // Numpy shapes are in (D, H, W) order after volume packing
    epiDims.D = epiBuf.shape[0]; epiDims.H = epiBuf.shape[1]; epiDims.W = epiBuf.shape[2];
    t1Dims.D = t1Buf.shape[0]; t1Dims.H = t1Buf.shape[1]; t1Dims.W = t1Buf.shape[2];

    auto epiVS = epiVoxelSize.unchecked<1>();
    auto t1VS = t1VoxelSize.unchecked<1>();
    VoxelSize epiVox = {epiVS(0), epiVS(1), epiVS(2)};
    VoxelSize t1Vox = {t1VS(0), t1VS(1), t1VS(2)};

    // Build QuadratureFilters
    QuadratureFilters filters;

    // Parametric filters (3, each 7x7x7 complex -> real + imag)
    for (int i = 0; i < 3; i++) {
        py::array_t<float> f = parametricFilters[i].cast<py::array_t<float>>();
        auto fb = f.request();
        // Input is complex interleaved: shape (343, 2) or (7,7,7) complex
        // We need to deinterleave real and imaginary parts
        int nelem = 343; // 7^3
        filters.linearReal[i].resize(nelem);
        filters.linearImag[i].resize(nelem);
        const float* fptr = static_cast<const float*>(fb.ptr);
        if (fb.ndim == 2 && fb.shape[1] == 2) {
            // (343, 2) layout
            for (int j = 0; j < nelem; j++) {
                filters.linearReal[i][j] = fptr[j * 2];
                filters.linearImag[i][j] = fptr[j * 2 + 1];
            }
        } else {
            // Assume already separated or flat real array
            // For complex numpy arrays, pybind11 gives us float view with 2x elements
            int total = fb.size;
            if (total == nelem * 2) {
                for (int j = 0; j < nelem; j++) {
                    filters.linearReal[i][j] = fptr[j * 2];
                    filters.linearImag[i][j] = fptr[j * 2 + 1];
                }
            } else {
                // Real only
                for (int j = 0; j < nelem; j++) {
                    filters.linearReal[i][j] = fptr[j];
                    filters.linearImag[i][j] = 0;
                }
            }
        }
    }

    // Nonparametric filters (6, each 7x7x7 complex)
    for (int i = 0; i < 6; i++) {
        py::array_t<float> f = nonparametricFilters[i].cast<py::array_t<float>>();
        auto fb = f.request();
        int nelem = 343;
        filters.nonlinearReal[i].resize(nelem);
        filters.nonlinearImag[i].resize(nelem);
        const float* fptr = static_cast<const float*>(fb.ptr);
        int total = fb.size;
        if (total >= nelem * 2) {
            for (int j = 0; j < nelem; j++) {
                filters.nonlinearReal[i][j] = fptr[j * 2];
                filters.nonlinearImag[i][j] = fptr[j * 2 + 1];
            }
        } else {
            for (int j = 0; j < nelem; j++) {
                filters.nonlinearReal[i][j] = fptr[j];
                filters.nonlinearImag[i][j] = 0;
            }
        }
    }

    // Projection tensors (6 x 6: m11,m12,m13,m22,m23,m33)
    for (int i = 0; i < 6; i++) {
        py::array_t<float> pt = projectionTensors[i].cast<py::array_t<float>>();
        auto ptb = pt.request();
        const float* pp = static_cast<const float*>(ptb.ptr);
        for (int j = 0; j < 6; j++) {
            filters.projectionTensors[i][j] = pp[j];
        }
    }

    // Filter directions (3 arrays of 6)
    for (int d = 0; d < 3; d++) {
        py::array_t<float> fd = filterDirections[d].cast<py::array_t<float>>();
        auto fdb = fd.request();
        const float* fdp = static_cast<const float*>(fdb.ptr);
        for (int i = 0; i < 6; i++) {
            if (d == 0) filters.filterDirectionsX[i] = fdp[i];
            else if (d == 1) filters.filterDirectionsY[i] = fdp[i];
            else filters.filterDirectionsZ[i] = fdp[i];
        }
    }

    // Run registration
    auto result = registerEPIT1(
        static_cast<const float*>(epiBuf.ptr), epiDims, epiVox,
        static_cast<const float*>(t1Buf.ptr), t1Dims, t1Vox,
        filters, numIterations, coarsestScale, mmZCut, verbose);

    // Convert results to numpy
    py::array_t<float> aligned({t1Dims.D, t1Dims.H, t1Dims.W});
    memcpy(aligned.mutable_data(), result.aligned.data(), result.aligned.size() * sizeof(float));

    py::array_t<float> interpolated({t1Dims.D, t1Dims.H, t1Dims.W});
    memcpy(interpolated.mutable_data(), result.interpolated.data(), result.interpolated.size() * sizeof(float));

    py::array_t<float> params(6);
    memcpy(params.mutable_data(), result.params.data(), 6 * sizeof(float));

    return py::make_tuple(aligned, interpolated, params);
}

static py::tuple py_registerT1MNI(
    py::array_t<float> t1Data, py::array_t<float> t1VoxelSize,
    py::array_t<float> mniData, py::array_t<float> mniVoxelSize,
    py::array_t<float> mniBrainData,
    py::array_t<float> mniMaskData,
    py::list parametricFilters,
    py::list nonparametricFilters,
    py::list projectionTensors,
    py::list filterDirections,
    int linearIterations, int nonlinearIterations, int coarsestScale, int mmZCut,
    int opencl_platform, int opencl_device, bool verbose)
{
    auto t1Buf = t1Data.request();
    auto mniBufReq = mniData.request();
    auto mniBrainBuf = mniBrainData.request();
    auto mniMaskBuf = mniMaskData.request();

    VolumeDims t1Dims, mniDims;
    t1Dims.D = t1Buf.shape[0]; t1Dims.H = t1Buf.shape[1]; t1Dims.W = t1Buf.shape[2];
    mniDims.D = mniBufReq.shape[0]; mniDims.H = mniBufReq.shape[1]; mniDims.W = mniBufReq.shape[2];

    auto t1VS = t1VoxelSize.unchecked<1>();
    auto mniVS = mniVoxelSize.unchecked<1>();
    VoxelSize t1Vox = {t1VS(0), t1VS(1), t1VS(2)};
    VoxelSize mniVox = {mniVS(0), mniVS(1), mniVS(2)};

    // Build filters (same as above)
    QuadratureFilters filters;
    for (int i = 0; i < 3; i++) {
        py::array_t<float> f = parametricFilters[i].cast<py::array_t<float>>();
        auto fb = f.request();
        int nelem = 343;
        filters.linearReal[i].resize(nelem);
        filters.linearImag[i].resize(nelem);
        const float* fptr = static_cast<const float*>(fb.ptr);
        int total = fb.size;
        if (total >= nelem * 2) {
            for (int j = 0; j < nelem; j++) {
                filters.linearReal[i][j] = fptr[j * 2];
                filters.linearImag[i][j] = fptr[j * 2 + 1];
            }
        } else {
            for (int j = 0; j < nelem; j++) {
                filters.linearReal[i][j] = fptr[j];
                filters.linearImag[i][j] = 0;
            }
        }
    }
    for (int i = 0; i < 6; i++) {
        py::array_t<float> f = nonparametricFilters[i].cast<py::array_t<float>>();
        auto fb = f.request();
        int nelem = 343;
        filters.nonlinearReal[i].resize(nelem);
        filters.nonlinearImag[i].resize(nelem);
        const float* fptr = static_cast<const float*>(fb.ptr);
        int total = fb.size;
        if (total >= nelem * 2) {
            for (int j = 0; j < nelem; j++) {
                filters.nonlinearReal[i][j] = fptr[j * 2];
                filters.nonlinearImag[i][j] = fptr[j * 2 + 1];
            }
        } else {
            for (int j = 0; j < nelem; j++) {
                filters.nonlinearReal[i][j] = fptr[j];
                filters.nonlinearImag[i][j] = 0;
            }
        }
    }
    for (int i = 0; i < 6; i++) {
        py::array_t<float> pt = projectionTensors[i].cast<py::array_t<float>>();
        auto ptb = pt.request();
        const float* pp = static_cast<const float*>(ptb.ptr);
        for (int j = 0; j < 6; j++) filters.projectionTensors[i][j] = pp[j];
    }
    for (int d = 0; d < 3; d++) {
        py::array_t<float> fd = filterDirections[d].cast<py::array_t<float>>();
        auto fdb = fd.request();
        const float* fdp = static_cast<const float*>(fdb.ptr);
        for (int i = 0; i < 6; i++) {
            if (d == 0) filters.filterDirectionsX[i] = fdp[i];
            else if (d == 1) filters.filterDirectionsY[i] = fdp[i];
            else filters.filterDirectionsZ[i] = fdp[i];
        }
    }

    auto result = registerT1MNI(
        static_cast<const float*>(t1Buf.ptr), t1Dims, t1Vox,
        static_cast<const float*>(mniBufReq.ptr), mniDims, mniVox,
        static_cast<const float*>(mniBrainBuf.ptr),
        static_cast<const float*>(mniMaskBuf.ptr),
        filters, linearIterations, nonlinearIterations, coarsestScale, mmZCut, verbose);

    int mniVol = mniDims.size();
    std::vector<ssize_t> shape = {mniDims.D, mniDims.H, mniDims.W};

    py::array_t<float> alignedLinear(shape);
    memcpy(alignedLinear.mutable_data(), result.alignedLinear.data(), mniVol * sizeof(float));

    py::array_t<float> alignedNonLinear(shape);
    memcpy(alignedNonLinear.mutable_data(), result.alignedNonLinear.data(), mniVol * sizeof(float));

    py::array_t<float> skullstripped(shape);
    memcpy(skullstripped.mutable_data(), result.skullstripped.data(), mniVol * sizeof(float));

    py::array_t<float> interpolated(shape);
    memcpy(interpolated.mutable_data(), result.interpolated.data(), mniVol * sizeof(float));

    py::array_t<float> params(12);
    memcpy(params.mutable_data(), result.params.data(), 12 * sizeof(float));

    py::array_t<float> dispX(shape), dispY(shape), dispZ(shape);
    if (!result.dispX.empty()) {
        memcpy(dispX.mutable_data(), result.dispX.data(), mniVol * sizeof(float));
        memcpy(dispY.mutable_data(), result.dispY.data(), mniVol * sizeof(float));
        memcpy(dispZ.mutable_data(), result.dispZ.data(), mniVol * sizeof(float));
    }

    return py::make_tuple(
        alignedLinear, alignedNonLinear, skullstripped, interpolated,
        params, dispX, dispY, dispZ);
}

PYBIND11_MODULE(metal_registration, m) {
    m.doc() = "Metal GPU-accelerated image registration (BROCCOLI)";

    m.def("registerEPIT1", &py_registerEPIT1,
        "Register EPI to T1 using parametric (phase-based) registration",
        py::arg("epi_data"), py::arg("epi_voxel_size"),
        py::arg("t1_data"), py::arg("t1_voxel_size"),
        py::arg("parametric_filters"),
        py::arg("nonparametric_filters"),
        py::arg("projection_tensors"),
        py::arg("filter_directions"),
        py::arg("num_iterations") = 20,
        py::arg("coarsest_scale") = 8,
        py::arg("mm_z_cut") = 0,
        py::arg("opencl_platform") = 0,
        py::arg("opencl_device") = 0,
        py::arg("verbose") = false);

    m.def("registerT1MNI", &py_registerT1MNI,
        "Register T1 to MNI using affine + nonlinear morphon registration",
        py::arg("t1_data"), py::arg("t1_voxel_size"),
        py::arg("mni_data"), py::arg("mni_voxel_size"),
        py::arg("mni_brain_data"), py::arg("mni_mask_data"),
        py::arg("parametric_filters"),
        py::arg("nonparametric_filters"),
        py::arg("projection_tensors"),
        py::arg("filter_directions"),
        py::arg("linear_iterations") = 10,
        py::arg("nonlinear_iterations") = 5,
        py::arg("coarsest_scale") = 4,
        py::arg("mm_z_cut") = 0,
        py::arg("opencl_platform") = 0,
        py::arg("opencl_device") = 0,
        py::arg("verbose") = false);
}
