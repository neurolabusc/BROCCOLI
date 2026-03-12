#!/bin/bash
# Build the BROCCOLI Python module on macOS (Apple Silicon)
set -e

BROCCOLI_GIT_DIRECTORY=$(git rev-parse --show-toplevel)
BROCCOLI_LIB_DIRECTORY=${BROCCOLI_GIT_DIRECTORY}/code/BROCCOLI_LIB
OPENCL_HEADER_DIRECTORY=$(xcrun --show-sdk-path)/System/Library/Frameworks/OpenCL.framework/Headers
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON_SO_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "=== Compiling broccoli_lib.cpp ==="
pushd ${BROCCOLI_LIB_DIRECTORY} > /dev/null
g++ -std=c++14 -DEIGEN_DONT_VECTORIZE \
    -I${OPENCL_HEADER_DIRECTORY} \
    -I${BROCCOLI_LIB_DIRECTORY} \
    -I${BROCCOLI_LIB_DIRECTORY}/clBLASMac \
    -O0 -g -fPIC -Wno-narrowing \
    -c -o broccoli_lib.o broccoli_lib.cpp -w
popd > /dev/null

echo "=== Running SWIG ==="
swig -c++ -python \
    -I${OPENCL_HEADER_DIRECTORY} \
    -I${BROCCOLI_LIB_DIRECTORY} \
    -I${BROCCOLI_LIB_DIRECTORY}/clBLASMac \
    broccoli_lib.i

echo "=== Compiling wrapper ==="
g++ -std=c++14 -fPIC -O0 -g -DEIGEN_DONT_VECTORIZE -Wno-narrowing \
    -I${OPENCL_HEADER_DIRECTORY} \
    -I${PYTHON_INCLUDE} \
    -I${NUMPY_INCLUDE} \
    -I${BROCCOLI_LIB_DIRECTORY} \
    -I${BROCCOLI_LIB_DIRECTORY}/clBLASMac \
    -o broccoli_lib_wrap.o -c broccoli_lib_wrap.cxx -w

echo "=== Linking shared library ==="
g++ -std=c++14 -fPIC -shared \
    -o _broccoli_base${PYTHON_SO_EXT} \
    -framework OpenCL \
    broccoli_lib_wrap.o ${BROCCOLI_LIB_DIRECTORY}/broccoli_lib.o \
    -undefined dynamic_lookup

echo "=== Installing to broccoli package ==="
cp _broccoli_base${PYTHON_SO_EXT} broccoli/
cp broccoli_base.py broccoli/

echo "=== Build complete ==="
echo "Run: python3 RegisterEPIT1.py --epi-file ../../register/EPI_brain.nii.gz --t1-file ../../register/t1_brain.nii.gz --filters-parametric-file ../../filters/filters_for_linear_registration.mat --filters-nonparametric-file ../../filters/filters_for_nonlinear_registration.mat"
