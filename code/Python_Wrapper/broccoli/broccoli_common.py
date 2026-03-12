from broccoli_base import *

import numpy
from nibabel import nifti1

import os

BROCCOLI_LIB_BASE = BROCCOLI_LIB

# DONE: Check that passing arrays to C method as 1D packed arrays is the same as passing arays using the 3D array wrappers
# DONE: Check that packing (packVolume) and unpacking (unpackOutputVolume) results in the original input array
# DONE: Transpose and reshape until the two conditions above are not met

def load_MNI_templates(mni_file, mni_brain_file = None, mni_brain_mask_file = None):
  if not mni_brain_file:
    derived = mni_file.replace('.nii', '_brain.nii')
    mni_brain_file = derived if os.path.exists(derived) else None
  if not mni_brain_mask_file:
    derived = mni_file.replace('.nii', '_brain_mask.nii')
    mni_brain_mask_file = derived if os.path.exists(derived) else None

  MNI_nni = nifti1.load(mni_file)
  MNI = MNI_nni.get_fdata()

  if mni_brain_file:
    MNI_brain = nifti1.load(mni_brain_file).get_fdata()
  else:
    # If no separate brain file, use the MNI file itself (assumed skull-stripped)
    MNI_brain = MNI

  if mni_brain_mask_file:
    MNI_brain_mask = nifti1.load(mni_brain_mask_file).get_fdata()
  else:
    # Derive mask from brain volume (non-zero voxels)
    MNI_brain_mask = (MNI_brain > 0).astype(numpy.float32)

  voxel_sizes = MNI_nni.header['pixdim'][1:4]

  return MNI, MNI_brain, MNI_brain_mask, voxel_sizes

def load_T1(t1_file):
  T1_nni = nifti1.load(t1_file)
  T1 = T1_nni.get_fdata()
  T1_voxel_sizes = T1_nni.header['pixdim'][1:4]
  return T1, T1_voxel_sizes
  
def load_EPI(epi_file, only_volume=True):
  EPI_nni = nifti1.load(epi_file)
  EPI = EPI_nni.get_fdata()
  if only_volume and EPI.ndim == 4:
    EPI = EPI[...,0]
  EPI_voxel_sizes = EPI_nni.header['pixdim'][1:4]
  return EPI, EPI_voxel_sizes

_pack_permutation = (2, 0, 1)
_pack_permutation_4d = (3, 2, 0, 1)

def _permute(permutation, array):
  n = len(array)
  return [array[permutation[i]] for i in range(n)] 

class BROCCOLI_LIB(BROCCOLI_LIB_BASE):
  def __init__(self, *args):
    BROCCOLI_LIB_BASE.__init__(self)
    """
      This is a hack to prevent Python from free()-ing arrays
      that have been packed and then passed to C
    """
    self._input_arrays = []
    
    if len(args) == 2:
      self.OpenCLInitiate(*args)
      
  def OpenCLInitiate(self, platform, device):
    # Set BROCCOLI_DIR so the C++ code can find kernel files in code/Kernels/
    broccoli_dir = os.environ.get('BROCCOLI_DIR')
    if not broccoli_dir:
      # Auto-detect: walk up from this file to find the repo root
      candidate = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      while candidate != os.path.dirname(candidate):
        if os.path.isdir(os.path.join(candidate, 'code', 'Kernels')):
          broccoli_dir = candidate + '/'
          break
        candidate = os.path.dirname(candidate)
    if broccoli_dir:
      os.environ['BROCCOLI_DIR'] = broccoli_dir
      self.SetWrapper(2)  # BASH=2, so it uses BROCCOLI_DIR env var
    BROCCOLI_LIB_BASE.OpenCLInitiate(self, platform, device)
    
  def SetEPIData(self, array, voxel_sizes):
    self.SetEPIHeight(array.shape[0])
    self.SetEPIWidth(array.shape[1])
    self.SetEPIDepth(array.shape[2])

    t = self.packVolume(array)
    self.SetInputEPIVolume(t)

    # VoxelSizeX maps to Width(j), Y to Height(i), Z to Depth(k)
    # NIfTI pixdim order is (i, j, k), so swap [0] and [1]
    self.SetEPIVoxelSizeX(voxel_sizes[1])
    self.SetEPIVoxelSizeY(voxel_sizes[0])
    self.SetEPIVoxelSizeZ(voxel_sizes[2])
    
  def SetT1Data(self, array, voxel_sizes):
    self.SetT1Height(array.shape[0])
    self.SetT1Width(array.shape[1])
    self.SetT1Depth(array.shape[2])
    self.SetT1Timepoints(array.shape[3] if array.ndim == 4 else 1)

    t = self.packVolume(array)
    self.SetInputT1Volume(t)

    # VoxelSizeX maps to Width(j), Y to Height(i), Z to Depth(k)
    self.SetT1VoxelSizeX(voxel_sizes[1])
    self.SetT1VoxelSizeY(voxel_sizes[0])
    self.SetT1VoxelSizeZ(voxel_sizes[2])
    
  def SetMNIData(self, array, voxel_sizes):
    self.SetMNIHeight(array.shape[0])
    self.SetMNIWidth(array.shape[1])
    self.SetMNIDepth(array.shape[2])

    t = self.packVolume(array)
    self.SetInputMNIVolume(t)

    # VoxelSizeX maps to Width(j), Y to Height(i), Z to Depth(k)
    self.SetMNIVoxelSizeX(voxel_sizes[1])
    self.SetMNIVoxelSizeY(voxel_sizes[0])
    self.SetMNIVoxelSizeZ(voxel_sizes[2])

  def SetfMRIData(self, array, voxel_sizes):
    self.SetEPIHeight(array.shape[0])
    self.SetEPIWidth(array.shape[1])
    self.SetEPIDepth(array.shape[2])
    self.SetEPITimepoints(array.shape[3])

    t = self.packVolume(array)
    self.SetInputfMRIVolumes(t)

    # VoxelSizeX maps to Width(j), Y to Height(i), Z to Depth(k)
    self.SetEPIVoxelSizeX(voxel_sizes[1])
    self.SetEPIVoxelSizeY(voxel_sizes[0])
    self.SetEPIVoxelSizeZ(voxel_sizes[2])

  def SetParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(3):
      args.append(self.packVolume(numpy.real(filters[i])))
      args.append(self.packVolume(numpy.imag(filters[i])))
    BROCCOLI_LIB_BASE.SetLinearImageRegistrationFilters(self, *args)

  def SetNonParametricImageRegistrationFilters(self, filters):
    args = []
    for i in range(6):
      args.append(self.packVolume(numpy.real(filters[i])))
      args.append(self.packVolume(numpy.imag(filters[i])))
    BROCCOLI_LIB_BASE.SetNonLinearImageRegistrationFilters(self, *args)
    
  def SetProjectionTensorMatrixFilters(self, filters):
    self.SetProjectionTensorMatrixFirstFilter(*filters[0])
    self.SetProjectionTensorMatrixSecondFilter(*filters[1])
    self.SetProjectionTensorMatrixThirdFilter(*filters[2])
    self.SetProjectionTensorMatrixFourthFilter(*filters[3])
    self.SetProjectionTensorMatrixFifthFilter(*filters[4])
    self.SetProjectionTensorMatrixSixthFilter(*filters[5])
    
  def packArray(self, array):
    return numpy.ascontiguousarray(array, dtype=numpy.float32)

  def packVolume(self, array):
    if len(array.shape) == 3:
      array = numpy.flipud(array)
      t = array.transpose(_pack_permutation)
    elif len(array.shape) == 4:
      array = numpy.flipud(array)
      t = array.transpose(_pack_permutation_4d)
    else:
      t = array
    t = self.packArray(t.flatten())
    self._input_arrays.append(t)
    return t

  def createOutputArray(self, shape, dtype=numpy.float32):
    return numpy.empty(shape=shape, dtype=dtype).flatten()

  def unpackOutputArray(self, array, shape):
    return array.reshape(shape)
    
  def unpackOutputVolume(self, array, shape = None):
    unpack = None
    if shape:
      if len(shape) == 3:
        t_shape = _permute(_pack_permutation, shape)
        unpack = numpy.argsort(_pack_permutation)
      elif len(shape) == 4:
        t_shape = _permute(_pack_permutation_4d, shape)
        unpack = numpy.argsort(_pack_permutation_4d)
      else:
        t_shape = shape
      array = self.unpackOutputArray(array, t_shape)
      array = numpy.fliplr(array)
      
    if unpack is not None:
      return array.transpose(unpack)
    else:
      return array


  def printSetupErrors(self):
    print("Get platform IDs error is %d" % self.GetOpenCLPlatformIDsError())
    print("Get device IDs error is %d" % self.GetOpenCLDeviceIDsError())
    print("Create context error is %d" % self.GetOpenCLCreateContextError())
    print("Get create context info error is %d" % self.GetOpenCLContextInfoError())
    print("Create command queue error is %d" % self.GetOpenCLCreateCommandQueueError())
    print("Create program error is %d" % self.GetOpenCLCreateProgramError())
    print("Build program error is %d" % self.GetOpenCLBuildProgramError())
    print("Get program build info error is %d" % self.GetOpenCLProgramBuildInfoError())
    
    numOpenKernels = self.GetNumberOfOpenCLKernels()
    createKernelErrors = self.GetOpenCLCreateKernelErrors()
    
    for i in range(numOpenKernels):
      error = createKernelErrors[i]
      if error:
        print("Run kernel error %d is %d" % (i, error))
        
  def printRunErrors(self):
    numOpenKernels = self.GetNumberOfOpenCLKernels()
    createBufferErrors = self.GetOpenCLCreateBufferErrors()
    runKernelErrors = self.GetOpenCLRunKernelErrors()
    
    for i in range(numOpenKernels):
      if createBufferErrors[i]:
        print("Create buffer error %d is %d" % (i, createBufferErrors[i]))
      if runKernelErrors[i]:
        print("Run kernel error %d is %d" % (i, runKernelErrors[i]))
