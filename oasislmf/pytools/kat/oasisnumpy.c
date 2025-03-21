#include <Python.h>
#include <complex.h>
#include <fcntl.h>
#include <listobject.h>
#include <numpy/arrayobject.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void *cast_value(void *value_ptr, int type_num)
{
    void *result = NULL;

    switch (type_num) {
        case NPY_INT32:
            result = malloc(sizeof(int));
            *(int *) result = *(int *) value_ptr;
            break;
        case NPY_FLOAT32:
            result = malloc(sizeof(float));
            *(float *) result = *(float *) value_ptr;
            break;
        case NPY_FLOAT64:
            result = malloc(sizeof(double));
            *(double *) result = *(double *) value_ptr;
            break;
        case NPY_BOOL:
            result = malloc(sizeof(int)); // Treat bool as int for printing
            *(int *) result = (*(npy_bool *) value_ptr) ? 1 : 0;
            break;
        default:
            break;
    }
    return result;
}

// Function to handle saving the NumPy array to a file or file-like object
static void _savefmttxt(FILE *fh, PyArrayObject *X, PyObject *fmt,
                        const char *newline)
{
    if (fh == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid file handle");
        return NULL;
    }

    int ndim = PyArray_NDIM(X);
    npy_intp *shape = PyArray_SHAPE(X);
    PyArray_Descr *dtype = PyArray_DTYPE(X);
    int ncols = shape[1];
    int nrows = shape[0];
    bool isnamedarray = false;
    PyObject *names;
    PyObject *fields;
    if (ndim != 2) {
        if (ndim == 1 && PyDataType_HASFIELDS(dtype)) {
            names = dtype->names;
            fields = dtype->fields;
            isnamedarray = true;
        }
        else {
            PyErr_Format(PyExc_ValueError,
                         "Expected a 2D array, got %dD array instead", ndim);
            return NULL;
        }
    }

    for (int i = 0; i < nrows; i++) {
        // Array of void ptrs to hold the arg values for each row to format
        // later
        void **arg_values = (void **) malloc(ncols * sizeof(void *));
        int *arg_types = (int *) malloc(ncols * sizeof(int));
        if (!arg_values) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
            return NULL;
        }

        if (isnamedarray) {
            if (!names) {
                PyErr_SetString(PyExc_ValueError,
                                "No named fields found for structured array");
                return NULL;
            }
            for (int j = 0; j < PyTuple_Size(names); j++) {
                // Get column dtype
                PyObject *key = PyTuple_GetItem(names, j);
                PyObject *field = PyDict_GetItem(dtype->fields, key);
                PyArray_Descr *field_dtype =
                    (PyArray_Descr *) PyTuple_GetItem(field, 0);

                // Get column value
                void *value_ptr =
                    PyArray_GETPTR1(X, i) + field_dtype->elsize * j;
                arg_values[j] = cast_value(value_ptr, field_dtype->type_num);
                arg_types[j] = field_dtype->type_num;
                printf("Val: %f\n", arg_values[j]);
                printf("Type: %i\n", arg_types[j]);
            }
        }
        else {
            for (int j = 0; j < ncols; j++) {
                void *value_ptr = PyArray_GETPTR2(X, i, j);
                arg_values[j] = cast_value(value_ptr, PyArray_TYPE(X));
                arg_types[j] = PyArray_TYPE(X);
            }
        }

        Py_ssize_t fmt_len = PyList_Size(fmt);
        for (int j = 0; j < fmt_len; j++) {
            if (j < fmt_len) {
                PyObject *fmt_item = PyList_GetItem(fmt, j);
                if (!PyUnicode_Check(fmt_item)) {
                    PyErr_SetString(PyExc_TypeError,
                                    "All elements in fmt must be strings");
                    return;
                }
                const char *fmt_str = PyUnicode_AsUTF8(fmt_item);

                // Now print the value with the respective format
                if (arg_values[j]) {
                    printf(fmt_str,
                           *(int *) arg_values[j]); // Assuming the cast to
                                                    // double for simplicity
                    printf(",");
                    printf(fmt_str,
                           *(float *) arg_values[j]); // Assuming the cast to
                                                      // double for simplicity
                }
                else {
                    PyErr_SetString(PyExc_ValueError, "Null value in array");
                    return;
                }

                // Add space between columns (unless it's the last column in the
                // row)
                if (j < ncols - 1) {
                    printf(" - ");
                }
            }
        }
        printf("\n");

        // Clean up memory
        if (isnamedarray) {
            for (int j = 0; j < PyTuple_Size(names); j++) {
                if (arg_values[j]) free(arg_values[j]);
            }
        }
        else {
            for (int j = 0; j < ncols; j++) {
                if (arg_values[j]) free(arg_values[j]);
            }
        }
        free(arg_values);
        free(arg_types);
    }

    fflush(fh); // TODO: why do I need a buffer here when C++ ktools does not
}

static PyObject *savefmttxt(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *fname;
    PyArrayObject *X;
    PyObject *fmt = PyList_New(0);
    const char *newline = "\n";

    static char *kwlist[] = {"fname", "X", "fmt", "newline", NULL};

    // Parse Python arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Os", kwlist, &fname, &X,
                                     &fmt, &newline)) {
        return NULL;
    }

    // Ensure X is a numpy array
    if (!PyArray_Check(X)) {
        PyErr_SetString(PyExc_TypeError, "X must be a numpy array");
        return NULL;
    }

    // Ensure fmt_list is a list
    if (!PyList_Check(fmt)) {
        PyErr_SetString(PyExc_TypeError, "fmt must be a list of strings");
        return NULL;
    }

    // Handle file argument
    FILE *fh = NULL;
    int own_fh = 0;

    // Case 1: fname is a string (file path)
    if (PyUnicode_Check(fname)) {
        const char *path = PyUnicode_AsUTF8(fname);
        fh = fopen(path, "w");
        if (fh == NULL) {
            PyErr_SetString(PyExc_OSError, "Failed to open file for writing");
            return NULL;
        }
        own_fh = 1;
    }
    // Case 2: fname is a file handle
    else if (PyObject_HasAttrString(fname, "write")) {
        PyObject *fileno = PyObject_CallMethod(fname, "fileno", NULL);
        if (fileno == NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "Invalid file handle (could not get fileno)");
            return NULL;
        }
        int fd = PyLong_AsLong(fileno);
        Py_DECREF(fileno);
        if (fd < 0) {
            PyErr_SetString(PyExc_ValueError, "Invalid file descriptor");
            return NULL;
        }
        fh = fdopen(fd, "w");
        if (fh == NULL) {
            PyErr_SetString(PyExc_OSError,
                            "Failed to convert file descriptor to FILE*");
            return NULL;
        }
    }
    // Case 3: fname is a Path object
    else {
        // Check if fname is an instance of pathlib.Path
        PyObject *pathlib = PyImport_ImportModule("pathlib");
        if (pathlib == NULL) {
            return NULL; // Import error, return
        }

        PyObject *pathlib_path = PyObject_GetAttrString(pathlib, "Path");
        Py_DECREF(pathlib);

        if (PyObject_IsInstance(fname, pathlib_path)) {
            const char *path = PyUnicode_AsUTF8(PyObject_Str(fname));
            fh = fopen(path, "w");
            if (fh == NULL) {
                PyErr_SetString(PyExc_OSError,
                                "Failed to open file for writing");
                return NULL;
            }
            own_fh = 1;
        }
        else {
            PyErr_SetString(
                PyExc_TypeError,
                "fname must be a string, file handle, or pathlib.Path");
            return NULL;
        }
    }

    // Call the C function to save the array to the file
    _savefmttxt(fh, X, fmt, newline);

    Py_RETURN_NONE;
};

// Module method definitions
static PyMethodDef OasisNumpyMethods[] = {
    {"savefmttxt", savefmttxt, METH_VARARGS | METH_KEYWORDS,
     "Save format text string from array to file"},
    {NULL, NULL, 0, NULL} // Sentinel value
};

// Module definition
static struct PyModuleDef oasisnumpy_ctools = {
    PyModuleDef_HEAD_INIT,
    "oasisnumpy_ctools",                                         // __name__
    "This is a module with certain numpy function written in C", // __doc__
    -1,               // interpreter size (default -1)
    OasisNumpyMethods // methods
};

// Module initialization
PyMODINIT_FUNC PyInit_oasisnumpy_ctools(void)
{
    import_array(); // Initialize numpy C API
    return PyModule_Create(&oasisnumpy_ctools);
}
