#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

static PyObject *EVCReaderError;

#define PANIC(A) { printf(A " %s:%i\n", __FILE__,__LINE__); exit(1); }
#define ASSERT(S, A) { if(!(S)) PANIC(A) }

typedef struct {
    uint16_t x;
    uint8_t y;
    uint8_t p;
} event_t;

#define NUM (512*512*64)
/* #define NUM (512) */

static PyObject *
evcreader_read(PyObject *self, PyObject *args) {
    const char *fn;
    long long n_frames;

    if (!PyArg_ParseTuple(args, "sL", &fn, &n_frames)) {
        return NULL;
    }

    event_t *evtbuf = (event_t*) malloc(sizeof(event_t)*NUM);

    FILE *f = fopen(fn, "rb");
    ASSERT(f, "can't open file");

    PyObject *ret = PyList_New(n_frames);
    ASSERT(ret, "can't create list");

    long long framebuf_cap = 1024*1024*1024;
    long long framebuf_size = 0;
    uint8_t *framebuf = (uint8_t*) malloc(framebuf_cap*sizeof(uint8_t));

    uint64_t evtbuf_size = fread_unlocked(evtbuf, sizeof(event_t), NUM, f);
    uint64_t off = 0;

    uint64_t cnt = 0;
    for (long long i = -1; i < n_frames;) {
        if (framebuf_size >= framebuf_cap) {
            framebuf_cap *= 2;
            framebuf = (uint8_t*) realloc(framebuf, framebuf_cap*sizeof(uint8_t));
            ASSERT(framebuf, "can't realloc framebuf");
        }
        if (cnt-off >= evtbuf_size) {
            off += evtbuf_size;
            evtbuf_size = fread_unlocked(evtbuf, sizeof(event_t), NUM, f);
        }

        event_t evt = evtbuf[cnt-off];

        if (evt.p == 255) {
            if (i >= 0) {
                npy_intp dims[2] = {framebuf_size/3, 3};
                PyObject *arr = PyArray_SimpleNew(2, dims, NPY_UINT8);

                ASSERT(arr, "can't create numpy array");

                uint8_t *data = (uint8_t*) PyArray_DATA((PyArrayObject*) arr);
                memcpy(data, framebuf, framebuf_size*sizeof(uint8_t));

                ASSERT(PyList_SetItem(ret, i, arr) == 0, "can't set list item");

                framebuf_size = 0;
            }
            ++i;
            /* if (i % 10000000 == 0) { */
            /*     printf("%lld frames loaded\n", i); */
            /* } */
        } else {
            framebuf[framebuf_size+0] = evt.x;
            framebuf[framebuf_size+1] = evt.y;
            framebuf[framebuf_size+2] = evt.p;
            framebuf_size += 3;
        }

        ++cnt;
    }
    fclose(f);
    free(evtbuf);
    printf("%lu\n", cnt);
    return ret;
}

static PyMethodDef EVCReaderMethods[] = {
    {"read",  evcreader_read, METH_VARARGS,
     "Read .evc file into a numpy array."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef evcreadermodule = {
    PyModuleDef_HEAD_INIT,
    "evcreader",  /* name of module */
    NULL,         /* module documentation, may be NULL */
    -1,           /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
    EVCReaderMethods
};

PyMODINIT_FUNC
PyInit_evcreader(void)
{
    PyObject *m;

    m = PyModule_Create(&evcreadermodule);
    if (m == NULL)
        return NULL;

    EVCReaderError = PyErr_NewException("evcreader.error", NULL, NULL);
    Py_XINCREF(EVCReaderError);
    if (PyModule_AddObject(m, "error", EVCReaderError) < 0) {
        Py_XDECREF(EVCReaderError);
        Py_CLEAR(EVCReaderError);
        Py_DECREF(m);
        return NULL;
    }

    import_array();

    return m;
}
