import cython
cimport numpy as np
import numpy as np1

cdef Py_ssize_t WIDTH = 240
cdef Py_ssize_t HEIGHT = 180
cdef np.float32_t WINDOW = 100./1000

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef gen_lnes(np.float32_t[:, :, ::1] lnes, 
              np.float32_t[:, :, ::1] lastfire, 
              np.float32_t ts, 
              np.float32_t window):
    cdef Py_ssize_t ymax, xmax
    ymax = lnes.shape[0]
    xmax = lnes.shape[1]
    cdef Py_ssize_t y, x, p
    for y in range(ymax):
        for x in range(xmax):
            for p in range(2):
                lnes[y, x, p] = max(0, (lastfire[y, x, p]-ts+window)/window)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def batch_concat(np.float32_t[:, :, :, ::1] batch not None, 
                 int batchidx, 
                 np.float32_t[:, :, ::1] lastfire not None, 
                 double lastframet, 
                 np.float32_t[::1] tstamp not None, 
                 np.int16_t[:, ::1] event not None):
    inputs = []
    cdef Py_ssize_t i
    cdef np.int16_t x, y, p
    cdef np.float32_t[:, :, ::1] lnes
    cdef np.float32_t ts
    cdef np.int64_t cnt
    lnes = np1.zeros((HEIGHT, WIDTH, 2), np1.float32)

    for i in range(len(tstamp)):
        x = event[i, 0]
        y = event[i, 1]
        p = event[i, 2]
        ts = tstamp[i]
        lastfire[y, x, p] = ts

        cnt += 1
        
        if ts >= lastframet+1/1000. and cnt > 10:
            # lnes = np1.maximum(0, (lastfire - ts + WINDOW))/WINDOW
            gen_lnes(lnes, lastfire, ts, WINDOW)
            lastframet = ts
            batch[batchidx] = lnes
            batchidx += 1
            cnt = 0
            if batchidx >= 16:
                inputs.append(batch)
                batch = np1.zeros_like(batch)
                batchidx = 0
    return batch, batchidx, inputs, lastfire, lastframet
