# Makefile for broccolini — standalone spatial normalization executable
#
# Usage:
#   make              # auto-detect (Metal on macOS, OpenCL elsewhere)
#   make BACKEND=metal
#   make BACKEND=opencl
#   make clean

# ---------- Auto-detect platform and backend ----------

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  BACKEND ?= metal
else
  BACKEND ?= opencl
endif

# ---------- Common settings ----------

CC       = clang
CXX      = clang++
CFLAGS   = -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-narrowing -w
LDFLAGS  =

TARGET   = broccolini
BUILDDIR = build

# Common C objects
C_SRCS    = main.c registration.c nifti_io.c
C_OBJS    = $(patsubst %.c,$(BUILDDIR)/%.o,$(C_SRCS))

# ---------- Metal backend (macOS only) ----------

ifeq ($(BACKEND),metal)
  DEFINES  = -DHAVE_METAL -DHAVE_ZLIB
  SHADER_PATH ?= $(CURDIR)/metal/shaders/registration.metal

  METAL_SRCS = metal/metal_backend.mm metal/metal_registration.mm
  METAL_OBJS = $(BUILDDIR)/metal_backend.o $(BUILDDIR)/metal_registration.o

  METAL_CXXFLAGS = $(CXXFLAGS) $(DEFINES) \
    -DMETAL_SHADER_DEFAULT_PATH="\"$(SHADER_PATH)\"" \
    -I. -Imetal

  FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
  LDFLAGS   += -lz $(FRAMEWORKS)

  ALL_OBJS = $(C_OBJS) $(METAL_OBJS)
endif

# ---------- OpenCL backend ----------

ifeq ($(BACKEND),opencl)
  DEFINES  = -DHAVE_OPENCL -DHAVE_ZLIB -DEIGEN_DONT_VECTORIZE

  # OpenCL headers location (macOS SDK or system)
  ifeq ($(UNAME),Darwin)
    OPENCL_HEADERS ?= $(shell xcrun --show-sdk-path)/System/Library/Frameworks/OpenCL.framework/Headers
    OPENCL_LDFLAGS  = -framework OpenCL
  else
    OPENCL_HEADERS ?= /usr/include/CL
    OPENCL_LDFLAGS  = -lOpenCL
  endif

  OPENCL_SRCS = opencl/opencl_backend.cpp opencl/broccoli_lib.cpp
  OPENCL_OBJS = $(BUILDDIR)/opencl_backend.o $(BUILDDIR)/broccoli_lib.o

  OPENCL_CXXFLAGS = $(CXXFLAGS) $(DEFINES) \
    -I. -Iopencl -I$(OPENCL_HEADERS)

  LDFLAGS += -lz $(OPENCL_LDFLAGS)
  ifeq ($(UNAME),Darwin)
    LDFLAGS += -framework Accelerate
  endif

  ALL_OBJS = $(C_OBJS) $(OPENCL_OBJS)
endif

# ---------- Rules ----------

.PHONY: all clean setup

all: setup $(TARGET)

setup:
	@mkdir -p $(BUILDDIR)
ifeq ($(BACKEND),opencl)
	@# Create symlink structure that BROCCOLI_LIB expects: code/Kernels/
	@mkdir -p opencl/code
	@ln -sfn ../kernels opencl/code/Kernels
endif

$(TARGET): $(ALL_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo ""
	@echo "=== Build successful: $(TARGET) ($(BACKEND) backend) ==="
	@echo ""
ifeq ($(BACKEND),opencl)
	@echo "NOTE: Set BROCCOLI_DIR before running:"
	@echo "  export BROCCOLI_DIR=$(CURDIR)/opencl/"
	@echo ""
endif
	@echo "Quick test:"
	@echo "  ./$(TARGET) -in ../register/t1_brain.nii.gz \\"
	@echo "    -ref ../register/MNI152_T1_2mm_brain.nii.gz \\"
	@echo "    -out /tmp/t1_aligned.nii.gz -verbose"

# Common C sources
$(BUILDDIR)/%.o: %.c
	$(CC) $(CFLAGS) $(DEFINES) -I. -c $< -o $@

# Metal backend
$(BUILDDIR)/metal_backend.o: metal/metal_backend.mm
	$(CXX) $(METAL_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/metal_registration.o: metal/metal_registration.mm
	$(CXX) $(METAL_CXXFLAGS) -c $< -o $@

# OpenCL backend
$(BUILDDIR)/opencl_backend.o: opencl/opencl_backend.cpp
	$(CXX) $(OPENCL_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/broccoli_lib.o: opencl/broccoli_lib.cpp
	$(CXX) $(OPENCL_CXXFLAGS) -fPIC -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)
	rm -f opencl/code/Kernels
	rmdir opencl/code 2>/dev/null || true
