#include <Python.h>
#include <complex.h>
#include <fcntl.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Function to handle saving the NumPy array to a file or file-like object
static void _savetxt(FILE *fh, PyArrayObject *X, const char *fmt,
                     const char *delimiter, const char *newline,
                     const char *header, const char *footer,
                     const char *comments)
{
    if (fh == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid file handle");
        return;
    }

    // Write header if provided
    if (strlen(header) > 0) {
        fprintf(fh, "%s%s%s", comments, header, newline);
    }

    // TODO: implement rest of savetxt

    // Write footer if provided
    if (strlen(footer) > 0) {
        fprintf(fh, "%s%s%s", comments, footer, newline);
    }
    fflush(fh); // TODO: why do I need a buffer here when C++ ktools does not
}

static PyObject *savetxt(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *fname;
    PyArrayObject *X;
    const char *fmt = "%.18e";
    const char *delimiter = " ";
    const char *newline = "\n";
    const char *header = "";
    const char *footer = "";
    const char *comments = "# ";

    static char *kwlist[] = {"fname",     "X",        "fmt",
                             "delimiter", "newline",  "header",
                             "footer",    "comments", NULL};

    // Parse Python arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ssssss", kwlist, &fname,
                                     &X, &fmt, &delimiter, &newline, &header,
                                     &footer, &comments)) {
        return NULL;
    }

    // Ensure X is a numpy array
    if (!PyArray_Check(X)) {
        PyErr_SetString(PyExc_TypeError, "X must be a numpy array");
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
    _savetxt(fh, X, fmt, delimiter, newline, header, footer, comments);

    Py_RETURN_NONE;
};

// Module method definitions
static PyMethodDef OasisNumpyMethods[] = {
    {"savetxt", savetxt, METH_VARARGS | METH_KEYWORDS,
     "Save an array to a text file."},
    {NULL, NULL, 0, NULL} // Sentinel value
};

// Module definition
static struct PyModuleDef oasisnumpy = {
    PyModuleDef_HEAD_INIT,
    "oasisnumpy",                                                // __name__
    "This is a module with certain numpy function written in C", // __doc__
    -1,               // interpreter size (default -1)
    OasisNumpyMethods // methods
};

// Module initialization
PyMODINIT_FUNC PyInit_oasisnumpy(void)
{
    import_array(); // Initialize numpy C API
    return PyModule_Create(&oasisnumpy);
}
