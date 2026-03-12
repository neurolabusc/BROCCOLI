%module broccoli_base
%{
  /* Includes the header in the wrapper code */
  #include "../BROCCOLI_LIB/broccoli_lib.h"
%}
 
/* Parse the header file to generate wrappers */

%{
  #define SWIG_FILE_WITH_INIT
%}

%include exception.i
%include "numpy.i"
%init %{
  import_array();
%}

%numpy_typemaps(float, NPY_FLOAT, int)

/* Accept any contiguous float numpy array as float* */
%typemap(in) float* {
    PyArrayObject* tmp = (PyArrayObject*)PyArray_FROM_OTF($input, NPY_FLOAT, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!tmp) SWIG_fail;
    $1 = (float*) PyArray_DATA(tmp);
    Py_DECREF(tmp);
}

/* Accept any contiguous int numpy array as int* (input) */
%typemap(in) int* {
    PyArrayObject* tmp = (PyArrayObject*)PyArray_FROM_OTF($input, NPY_INT, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!tmp) SWIG_fail;
    $1 = (int*) PyArray_DATA(tmp);
    Py_DECREF(tmp);
}

%typemap(out) int *
{
    $result = PyList_New(200);
    for (int i = 0; i < 200; ++i) {
        PyList_SetItem($result, i, PyLong_FromLong($1[i]));
    }
}

%ignore Coords3D::operator[];
%include "../BROCCOLI_LIB/broccoli_constants.h"
%include "../BROCCOLI_LIB/broccoli_lib.h"

typedef unsigned int cl_uint;

%extend Coords3D
{
    int __getitem__(int i) const
    {
        if (i < 0 || i > 2)
        {
            SWIG_Error(SWIG_IndexError, "index must be between 0 and 2");
            return 0;
        }
        return (*$self)[i];
    }
    
    void __setitem__(int i, int v)
    {
        if (i < 0 || i > 2)
        {
            SWIG_Error(SWIG_IndexError, "index must be between 0 and 2");
            return;
        }
        (*$self)[i] = v;
    }
    
    const char* __repr__() const
    {
        char* buf = (char*)malloc(64);
        sprintf(buf, "Coords3D(%d, %d, %d)", (*$self)[0], (*$self)[1], (*$self)[2]);
        return buf;
    }
}
