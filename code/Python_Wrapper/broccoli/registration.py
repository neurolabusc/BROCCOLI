from . import broccoli_common as broccoli

import numpy

def plotVolume(data, sliceYrel, sliceZrel):
  import matplotlib.pyplot as plot
  import matplotlib.cm as cm
  sliceY = int(round(sliceYrel * data.shape[0]))

  # Data is ordered [y][x][z]
  plot.imshow(numpy.flipud(data[sliceY].transpose()), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

  sliceZ = int(round(sliceZrel * data.shape[2])) - 1

  plot.imshow(numpy.fliplr(data[:,:,sliceZ]), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

def registerEPIT1(
    h_EPI_Data,
    h_EPI_Voxel_Sizes,
    h_T1_Data,          # Array
    h_T1_Voxel_Sizes,   # 3 elements
    h_Quadrature_Filter_Parametric_Registration,            # 3 elements, complex arrays
    h_Quadrature_Filter_NonParametric_Registration,         # 6 elements, complex arrays
    h_Projection_Tensor,             # 6 elements
    h_Filter_Directions,             # 3 elements
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION,     # int
    COARSEST_SCALE,         # int
    MM_EPI_Z_CUT,            # int
    OPENCL_PLATFORM,        # int
    OPENCL_DEVICE,          # int
    show_results = False,
  ):

  BROCCOLI = broccoli.BROCCOLI_LIB()
  print("Initializing OpenCL...")

  BROCCOLI.OpenCLInitiate(OPENCL_PLATFORM, OPENCL_DEVICE)
  ok = BROCCOLI.GetOpenCLInitiated()

  if ok == 0:
    BROCCOLI.printSetupErrors()
    print("OpenCL initialization failed, aborting")
    return

  print("OpenCL initialization successful, proceeding...")

  ## Set constants
  T1_DATA_SHAPE = h_T1_Data.shape
  h_T1_Data_orig = h_T1_Data  # Keep original layout for masking

  ## Make all arrays contiguous
  h_T1_Data = BROCCOLI.packArray(h_T1_Data)
  h_EPI_Data = BROCCOLI.packArray(h_EPI_Data)

  h_EPI_Voxel_Sizes = [float(i) for i in h_EPI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [float(i) for i in h_T1_Voxel_Sizes]

  ## Pass input parameters to BROCCOLI
  ## For EPI-T1 registration via PerformRegistrationTwoVolumesWrapper:
  ## - EPI is the "T1" input (volume to be registered)
  ## - T1 is the "MNI" reference (target space)
  print("Setting up input parameters...")

  print("EPI size is %s" % ' x '.join([str(i) for i in h_EPI_Data.shape]))
  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))

  # Set EPI as the input volume (T1 slot) and T1 as the reference (MNI slot)
  BROCCOLI.SetT1Data(h_EPI_Data, h_EPI_Voxel_Sizes)
  BROCCOLI.SetMNIData(h_T1_Data, h_T1_Voxel_Sizes)

  # Use the T1 brain as both the MNI brain volume and mask
  BROCCOLI.SetInputMNIBrainVolume(BROCCOLI.packVolume(h_T1_Data))
  BROCCOLI.SetInputMNIBrainMask(BROCCOLI.packVolume((h_T1_Data > 0).astype(numpy.float32)))

  BROCCOLI.SetInterpolationMode(broccoli.LINEAR)
  BROCCOLI.SetNumberOfIterationsForLinearImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonLinearImageRegistration(0)

  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[BROCCOLI.packArray(a) for a in h_Filter_Directions])

  BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE)
  BROCCOLI.SetMMT1ZCUT(MM_EPI_Z_CUT)

  ## Set up output parameters
  print("Setting up output parameters...")

  h_Aligned_EPI_Volume = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeLinear(h_Aligned_EPI_Volume)

  h_Interpolated_EPI_Volume = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_EPI_Volume)

  h_Registration_Parameters = BROCCOLI.createOutputArray(12)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters)

  h_Phase_Differences = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)

  h_Phase_Certainties = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)

  h_Phase_Gradients = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)

  # Need dummy outputs for the wrapper
  h_Aligned_NL = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeNonLinear(h_Aligned_NL)

  h_Skullstripped = BROCCOLI.createOutputArray(T1_DATA_SHAPE)
  BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped)

  h_Slice_Sums = BROCCOLI.createOutputArray(T1_DATA_SHAPE[2])
  BROCCOLI.SetOutputSliceSums(h_Slice_Sums)

  h_Top_Slice = BROCCOLI.createOutputArray(1)
  BROCCOLI.SetOutputTopSlice(h_Top_Slice)

  h_A_Matrix = BROCCOLI.createOutputArray(12 * 12)
  BROCCOLI.SetOutputAMatrix(h_A_Matrix)

  h_h_Vector = BROCCOLI.createOutputArray(12)
  BROCCOLI.SetOutputHVector(h_h_Vector)

  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationTwoVolumesWrapper()

  print("Registration done, unpacking output volumes...")

  h_Interpolated_EPI_Volume = BROCCOLI.unpackOutputVolume(h_Interpolated_EPI_Volume, T1_DATA_SHAPE)
  h_Aligned_EPI_Volume = BROCCOLI.unpackOutputVolume(h_Aligned_EPI_Volume, T1_DATA_SHAPE)

  # Mask output volumes: zero out voxels outside the reference (T1) brain
  ref_mask = (h_T1_Data_orig > 0).astype(numpy.float32)
  h_Aligned_EPI_Volume = h_Aligned_EPI_Volume * ref_mask
  h_Interpolated_EPI_Volume = h_Interpolated_EPI_Volume * ref_mask

  # Extract the 6 rigid-body parameters from the 12 affine parameters
  h_Registration_Parameters_6 = h_Registration_Parameters[:6]
  print(h_Registration_Parameters_6)

  if show_results:
    plot_results = (
      h_Interpolated_EPI_Volume,
      h_Aligned_EPI_Volume,
      h_T1_Data,
    )

    sliceY = 0.45
    sliceZ = 0.62

    import matplotlib.pyplot as plot
    for r in plot_results:
      plotVolume(r, sliceY, sliceZ)
    plot.close()
    plot.show()

  return (h_Aligned_EPI_Volume, h_Interpolated_EPI_Volume,
          h_Registration_Parameters_6)

def registerT1MNI(
    h_T1_Data,          # Array
    h_T1_Voxel_Sizes,   # 3 elements
    h_MNI_Data,         # Array
    h_MNI_Voxel_Sizes,   # 3 elements
    h_MNI_Brain,        # brain volume
    h_MNI_Brain_Mask,   # brain mask
    h_Quadrature_Filter_Parametric_Registration,            # 3 elements, complex arrays
    h_Quadrature_Filter_NonParametric_Registration,         # 6 elements, complex arrays
    h_Projection_Tensor,             # 6 elements
    h_Filter_Directions,             # 3 elements
    NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION,     # int
    NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION,  # int
    COARSEST_SCALE,         # int
    MM_T1_Z_CUT,            # int
    OPENCL_PLATFORM,        # int
    OPENCL_DEVICE,          # int
    show_results = False,
  ):

  BROCCOLI = broccoli.BROCCOLI_LIB()
  print("Initializing OpenCL...")

  BROCCOLI.OpenCLInitiate(OPENCL_PLATFORM, OPENCL_DEVICE)
  ok = BROCCOLI.GetOpenCLInitiated()

  if ok == 0:
    BROCCOLI.printSetupErrors()
    print("OpenCL initialization failed, aborting")
    return

  print("OpenCL initialization successful, proceeding...")

  ## Set constants
  NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS = 12

  MNI_DATA_SHAPE = h_MNI_Data.shape
  T1_DATA_SHAPE = h_T1_Data.shape

  ## Make all arrays contiguous
  h_T1_Data = BROCCOLI.packArray(h_T1_Data)
  h_MNI_Data = BROCCOLI.packArray(h_MNI_Data)
  h_MNI_Brain = BROCCOLI.packArray(h_MNI_Brain)
  h_MNI_Brain_Mask = BROCCOLI.packArray(h_MNI_Brain_Mask)

  h_MNI_Voxel_Sizes = [float(i) for i in h_MNI_Voxel_Sizes]
  h_T1_Voxel_Sizes = [float(i) for i in h_T1_Voxel_Sizes]

  ## Pass input parameters to BROCCOLI
  print("Setting up input parameters...")

  print("T1 size is %s" % ' x '.join([str(i) for i in h_T1_Data.shape]))
  print("MNI size is %s" % ' x '.join([str(i) for i in h_MNI_Data.shape]))

  BROCCOLI.SetT1Data(h_T1_Data, h_T1_Voxel_Sizes)
  BROCCOLI.SetMNIData(h_MNI_Data, h_MNI_Voxel_Sizes)

  BROCCOLI.SetInputMNIBrainVolume(BROCCOLI.packVolume(h_MNI_Brain))
  BROCCOLI.SetInputMNIBrainMask(BROCCOLI.packVolume(h_MNI_Brain_Mask))

  BROCCOLI.SetInterpolationMode(broccoli.LINEAR)
  BROCCOLI.SetNumberOfIterationsForLinearImageRegistration(NUMBER_OF_ITERATIONS_FOR_PARAMETRIC_IMAGE_REGISTRATION)
  BROCCOLI.SetNumberOfIterationsForNonLinearImageRegistration(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION)

  BROCCOLI.SetImageRegistrationFilterSize(h_Quadrature_Filter_Parametric_Registration[0][0].shape[0])
  BROCCOLI.SetParametricImageRegistrationFilters(h_Quadrature_Filter_Parametric_Registration)
  BROCCOLI.SetNonParametricImageRegistrationFilters(h_Quadrature_Filter_NonParametric_Registration)

  BROCCOLI.SetProjectionTensorMatrixFilters(h_Projection_Tensor)
  BROCCOLI.SetFilterDirections(*[BROCCOLI.packArray(a) for a in h_Filter_Directions])

  BROCCOLI.SetCoarsestScaleT1MNI(COARSEST_SCALE)
  BROCCOLI.SetMMT1ZCUT(MM_T1_Z_CUT)
  BROCCOLI.SetDoSkullstrip(True)
  BROCCOLI.SetSaveDisplacementField(NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION > 0)

  ## Set up output parameters
  print("Setting up output parameters...")

  h_Aligned_T1_Volume = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeLinear(h_Aligned_T1_Volume)

  h_Aligned_T1_Volume_NonParametric = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputAlignedT1VolumeNonLinear(h_Aligned_T1_Volume_NonParametric)

  h_Skullstripped_T1_Volume = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputSkullstrippedT1Volume(h_Skullstripped_T1_Volume)

  h_Interpolated_T1_Volume = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputInterpolatedT1Volume(h_Interpolated_T1_Volume)

  h_Registration_Parameters = BROCCOLI.createOutputArray(12)
  BROCCOLI.SetOutputT1MNIRegistrationParameters(h_Registration_Parameters)

  h_Phase_Differences = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputPhaseDifferences(h_Phase_Differences)

  h_Phase_Certainties = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputPhaseCertainties(h_Phase_Certainties)

  h_Phase_Gradients = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
  BROCCOLI.SetOutputPhaseGradients(h_Phase_Gradients)

  h_Slice_Sums = BROCCOLI.createOutputArray(h_MNI_Data.shape[2])
  BROCCOLI.SetOutputSliceSums(h_Slice_Sums)

  h_Top_Slice = BROCCOLI.createOutputArray(1)
  BROCCOLI.SetOutputTopSlice(h_Top_Slice)

  h_A_Matrix = BROCCOLI.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS * NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputAMatrix(h_A_Matrix)

  h_h_Vector = BROCCOLI.createOutputArray(NUMBER_OF_AFFINE_IMAGE_REGISTRATION_PARAMETERS)
  BROCCOLI.SetOutputHVector(h_h_Vector)

  # Displacement field outputs (only read back when WRITE_DISPLACEMENT_FIELD is set)
  if NUMBER_OF_ITERATIONS_FOR_NONPARAMETRIC_IMAGE_REGISTRATION > 0:
    h_Displacement_Field_X = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
    h_Displacement_Field_Y = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
    h_Displacement_Field_Z = BROCCOLI.createOutputArray(MNI_DATA_SHAPE)
    BROCCOLI.SetOutputDisplacementField(h_Displacement_Field_X, h_Displacement_Field_Y, h_Displacement_Field_Z)
  else:
    h_Displacement_Field_X = None
    h_Displacement_Field_Y = None
    h_Displacement_Field_Z = None

  ## Perform registration
  print("Performing registration...")
  BROCCOLI.PerformRegistrationTwoVolumesWrapper()

  print("Registration done, unpacking output volumes...")

  h_Aligned_T1_Volume = BROCCOLI.unpackOutputVolume(h_Aligned_T1_Volume, MNI_DATA_SHAPE)
  h_Interpolated_T1_Volume = BROCCOLI.unpackOutputVolume(h_Interpolated_T1_Volume, MNI_DATA_SHAPE)
  h_Aligned_T1_Volume_NonParametric = BROCCOLI.unpackOutputVolume(h_Aligned_T1_Volume_NonParametric, MNI_DATA_SHAPE)
  h_Skullstripped_T1_Volume = BROCCOLI.unpackOutputVolume(h_Skullstripped_T1_Volume, MNI_DATA_SHAPE)
  h_Phase_Differences = BROCCOLI.unpackOutputVolume(h_Phase_Differences, MNI_DATA_SHAPE)
  h_Phase_Certainties = BROCCOLI.unpackOutputVolume(h_Phase_Certainties, MNI_DATA_SHAPE)
  h_Phase_Gradients = BROCCOLI.unpackOutputVolume(h_Phase_Gradients, MNI_DATA_SHAPE)

  if h_Displacement_Field_X is not None:
    h_Displacement_Field_X = BROCCOLI.unpackOutputVolume(h_Displacement_Field_X, MNI_DATA_SHAPE)
    h_Displacement_Field_Y = BROCCOLI.unpackOutputVolume(h_Displacement_Field_Y, MNI_DATA_SHAPE)
    h_Displacement_Field_Z = BROCCOLI.unpackOutputVolume(h_Displacement_Field_Z, MNI_DATA_SHAPE)

  print(h_Registration_Parameters)

  if show_results:
    plot_results = (
      h_Interpolated_T1_Volume,
      h_Aligned_T1_Volume,
      h_MNI_Brain,
      h_Aligned_T1_Volume_NonParametric,
    )

    sliceY = 0.45
    sliceZ = 0.47

    import matplotlib.pyplot as plot
    for r in plot_results:
      plotVolume(r, sliceY, sliceZ)
    plot.close()
    plot.show()

  return (h_Aligned_T1_Volume, h_Aligned_T1_Volume_NonParametric, h_Skullstripped_T1_Volume, h_Interpolated_T1_Volume,
          h_Registration_Parameters, h_Phase_Differences, h_Phase_Certainties, h_Phase_Gradients, h_Slice_Sums, h_Top_Slice, h_A_Matrix, h_h_Vector,
          h_Displacement_Field_X, h_Displacement_Field_Y, h_Displacement_Field_Z)
