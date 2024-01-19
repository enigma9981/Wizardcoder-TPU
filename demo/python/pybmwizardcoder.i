
%module pybmwizardcoder

%{
    #define SWIG_FILE_WITH_INIT
    #include "wizardcoder_c.h"
%}

%include "numpy.i"

%init %{
import_array();
%}
%apply (int* IN_ARRAY1, int DIM1) { (int* devids, int num_device)};

%include "wizardcoder_c.h"