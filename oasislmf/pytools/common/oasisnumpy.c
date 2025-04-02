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

char *cast_value(void *value_ptr, int type_num, const char *fmt_str)
{
    static char buffer[64]; // Fixed buffer for formatted output

    if (!value_ptr || !fmt_str) {
        return NULL;
    }

    switch (type_num) {
        case NPY_INT32:
            snprintf(buffer, sizeof(buffer), fmt_str, *(int *) value_ptr);
            break;
        case NPY_FLOAT32:
            snprintf(buffer, sizeof(buffer), fmt_str, *(float *) value_ptr);
            break;
        case NPY_FLOAT64:
            snprintf(buffer, sizeof(buffer), fmt_str, *(double *) value_ptr);
            break;
        case NPY_BOOL:
            snprintf(buffer, sizeof(buffer), fmt_str,
                     (*(npy_bool *) value_ptr) ? 1 : 0);
            break;
        default:
            return NULL; // Unsupported type
    }

    return buffer;
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
        if (isnamedarray) {
            if (!names) {
                PyErr_SetString(PyExc_ValueError,
                                "No named fields found for structured array");
                return NULL;
            }
            Py_ssize_t fmt_len = PyList_Size(fmt);
            if (fmt_len != PyTuple_Size(names)) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Format string length does not match number of columns");
            }
            for (int j = 0; j < fmt_len; j++) {
                // Get column dtype
                PyObject *key = PyTuple_GetItem(names, j);
                PyObject *field = PyDict_GetItem(dtype->fields, key);
                PyArray_Descr *field_dtype =
                    (PyArray_Descr *) PyTuple_GetItem(field, 0);

                // Get column value
                void *value_ptr =
                    PyArray_GETPTR1(X, i) + field_dtype->elsize * j;
                int type = field_dtype->type_num;
                PyObject *fmt_item = PyList_GetItem(fmt, j);
                const char *fmt_str = PyUnicode_AsUTF8(fmt_item);
                const char *towrite = cast_value(value_ptr, type, fmt_str);
                fwrite(towrite, 1, strlen(towrite), fh);
                if (j < fmt_len - 1) {
                    fprintf(fh, ",");
                }
            }
        }
        else {
            Py_ssize_t fmt_len = PyList_Size(fmt);
            if (fmt_len != ncols) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Format string length does not match number of columns");
            }
            for (int j = 0; j < ncols; j++) {
                void *value_ptr = PyArray_GETPTR2(X, i, j);
                int type = PyArray_TYPE(X);
                PyObject *fmt_item = PyList_GetItem(fmt, j);
                const char *fmt_str = PyUnicode_AsUTF8(fmt_item);

                const char *towrite = cast_value(value_ptr, type, fmt_str);
                fwrite(towrite, 1, strlen(towrite), fh);
                if (j < fmt_len - 1) {
                    fputs(",", fh);
                }
            }
        }
        fputs("\n", fh);
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

PyObject *_fmtarray(PyArrayObject *row, PyObject *fmt, const char *newline)
{
    if (!row || !fmt || !newline) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments");
        return NULL;
    }

    int ndim = PyArray_NDIM(row);
    npy_intp *shape = PyArray_SHAPE(row);
    PyArray_Descr *dtype = PyArray_DTYPE(row);
    int ncols = shape[0];
    bool isnamedarray = PyDataType_HASFIELDS(dtype);
    PyObject *names = NULL;
    PyObject *fields = NULL;

    if (ndim != 1) {
        PyErr_Format(PyExc_ValueError,
                     "Expected a 1D array, got %dD array instead", ndim);
        return NULL;
    }

    if (isnamedarray) {
        names = dtype->names;
        fields = dtype->fields;
        ncols = PyTuple_Size(names);
        if (!names) {
            PyErr_SetString(PyExc_ValueError,
                            "No named fields found for structured array");
            return NULL;
        }
    }

    // Initial buffer size
    size_t bufsize = 256;
    char *buffer = (char *) malloc(bufsize);
    if (!buffer) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    buffer[0] = '\0'; // Initialize buffer as empty string

    for (int j = 0; j < ncols; j++) {
        char temp_buffer[64] = {0}; // Temporary buffer for formatting

        void *value_ptr = PyArray_GETPTR1(row, j);
        int type = PyArray_TYPE(row);
        PyObject *fmt_item = PyList_GetItem(fmt, j);
        const char *fmt_str = PyUnicode_AsUTF8(fmt_item);
        const char *towrite = cast_value(value_ptr, type, fmt_str);

        printf("TEST %s, %s, %d\n", towrite, fmt_str, type);

        strcat(temp_buffer, towrite);
        if (j < ncols - 1) strcat(temp_buffer, ","); // Add comma between values

        // Expand buffer if needed
        size_t new_len = strlen(buffer) + strlen(temp_buffer) + 1;
        if (new_len > bufsize) {
            bufsize *= 2;
            buffer = (char *) realloc(buffer, bufsize);
            if (!buffer) {
                PyErr_SetString(PyExc_MemoryError,
                                "Failed to reallocate memory");
                return NULL;
            }
        }
        strcat(buffer, temp_buffer);
    }

    strcat(buffer, newline); // Append newline character
    PyObject *result = PyUnicode_FromString(buffer);
    free(buffer);
    return result; // Caller must free() this
}

static PyObject *fmtarray(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *X;
    PyObject *fmt = PyList_New(0);
    const char *newline = "\n";

    static char *kwlist[] = {"X", "fmt", "newline", NULL};

    // Parse Python arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Os", kwlist, &X, &fmt,
                                     &newline)) {
        return NULL;
    }

    // Check if X is a NumPy array or a structured row (`numpy.void`)
    if (!PyArray_Check(X) &&
        !PyObject_IsInstance(X, (PyObject *) &PyVoidArrType_Type)) {
        PyErr_SetString(
            PyExc_TypeError,
            "X must be a NumPy array or a structured row (numpy.void)");
        return NULL;
    }

    // Ensure fmt_list is a list
    if (!PyList_Check(fmt)) {
        PyErr_SetString(PyExc_TypeError, "fmt must be a list of strings");
        return NULL;
    }

    PyArrayObject *array;
    if (PyArray_Check(X)) {
        array = (PyArrayObject *) X; // Normal case: X is already a NumPy array
    }
    else {
        // TODO: FIX THIS CASE
        PyArray_Descr *dtype = NULL;
        if (PyArray_CheckScalar(X)) {
            // If X is a scalar, get the dtype
            dtype = PyArray_DescrFromScalar(X);
            if (!dtype) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Failed to get dtype from scalar");
                return NULL;
            }

            // Create a 1D array from the scalar, which will preserve the dtype
            // and type
            npy_intp dims[1] = {1}; // Size 1 array
            array = (PyArrayObject *) PyArray_NewFromDescr(
                &PyArray_Type, dtype, 1, dims, NULL,
                PyDataMem_NEW(dtype->elsize), NPY_ARRAY_CARRAY, NULL);

            if (!array) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Failed to create 1D NumPy array from scalar");
                return NULL;
            }

            // Copy the scalar data into the newly created array
            memcpy(PyArray_DATA(array), PyArray_DATA((PyArrayObject *) X),
                   dtype->elsize);
        }
        else {
            // If X is neither a scalar nor a NumPy array, it's an invalid input
            PyErr_SetString(PyExc_TypeError,
                            "Expected a NumPy array or scalar");
            return NULL;
        }
    }

    // Call the C function to save the array to the file
    PyObject *result = _fmtarray(array, fmt, newline);
    return result;
};

// Module method definitions
static PyMethodDef OasisNumpyMethods[] = {
    {"savefmttxt", savefmttxt, METH_VARARGS | METH_KEYWORDS,
     "Save format text string from array to file"},
    {"fmtarray", fmtarray, METH_VARARGS | METH_KEYWORDS,
     "Format array to string"},
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
