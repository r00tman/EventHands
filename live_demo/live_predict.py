import numpy as np
import scipy.ndimage.morphology as morph
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dv import AedatFile, NetworkNumpyEventPacketInput
import cv2 as cv
import ffmpeg
import time
import struct
import os
from numba import jit, njit
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from model import MNISTModel
from live_cython import batch_concat

WIDTH, HEIGHT = 240, 180
WINDOW = 100/1000

fastcntr = 0

def stream_events(fps=1000):
    global fastcntr
    while True:
        try:
            starttime = time.time()
            start = None
            prevts = 0
            f = NetworkNumpyEventPacketInput(address='127.0.0.1', port=7777)
            for e in f:
                # print(e['timestamp'])
                pass
                if start is None:
                    start = e[0]['timestamp']
                    prevts = start
                ts = e['timestamp']-start

                xs, ys, ps = e['x'], e['y'], e['polarity']

                yield (ts.astype(np.float32)/1000000).astype(np.float32), np.c_[xs, ys, ps].astype(np.int16)

                if ts[-1] % 1000000 < (prevts-start) % 1000000:
                    now = time.time()-starttime
                    #print(ts[-1]/1000000, now, now-ts[-1]/1000000, 'fasts:', fastcntr)
                    print(ts[-1]/1000000, now, now-ts[-1]/1000000)
                    fastcntr = 0
                prevts = e[-1]['timestamp']
        except struct.error as e:
            print(e)
            continue
        except ConnectionRefusedError as e:
            print(e)
            continue

        return events


def main():
    global fastcntr
    fifopath = "/tmp/preds"
    if not os.path.exists(fifopath):
        os.mkfifo(fifopath, 0o600)
    fifowrite = open(fifopath, 'wb', 0)

    FPS = 1000
    my_filter = KalmanFilter(dim_x=2*12, dim_z=12)
    my_filter.x = np.zeros(24)
    my_filter.F = np.zeros((2*12, 2*12))
    my_filter.H = np.zeros((12, 2*12))
    my_filter.R *= 2.0
    fast_R = my_filter.R.copy()
    fast_Q = Q_discrete_white_noise(2, 1./FPS, .3, block_size=12)
    slow_Q = Q_discrete_white_noise(2, 1./FPS, .03, block_size=12)

    my_filter.Q = fast_Q
    for i in range(12):
        my_filter.F[2*i, 2*i] = 1.
        my_filter.F[2*i, 2*i+1] = 1.
        my_filter.F[2*i+1, 2*i] = 0.
        my_filter.F[2*i+1, 2*i+1] = 1.

        my_filter.H[i, 2*i] = 1.

    my_filter_slow = KalmanFilter(dim_x=2*12, dim_z=12)
    my_filter_slow.x = np.zeros(24)
    my_filter_slow.F = np.zeros((2*12, 2*12))
    my_filter_slow.H = np.zeros((12, 2*12))
    my_filter_slow.R *= 5.0
    slow_R = my_filter_slow.R.copy()
    my_filter_slow.Q = slow_Q
    for i in range(12):
        my_filter_slow.F[2*i, 2*i] = 1.
        my_filter_slow.F[2*i, 2*i+1] = 1.
        my_filter_slow.F[2*i+1, 2*i] = 0.
        my_filter_slow.F[2*i+1, 2*i+1] = 1.

        my_filter_slow.H[i, 2*i] = 1.

    model = MNISTModel.load_from_checkpoint(checkpoint_path='model.ckpt').cuda().half()
    model.eval()
    stream = stream_events()
    cv.namedWindow('stream')
    prevtime = time.time()
    lastfire = np.zeros((HEIGHT, WIDTH, 2), np.float32)-100
    batch = np.zeros((16, HEIGHT, WIDTH, 2), np.float32)
    batchidx = 0
    lastframet = -100
    pred = None
    for tstamp, event in stream:
        if tstamp[0] < lastframet:
            lastframet = -100
            lastfire = np.zeros((HEIGHT, WIDTH, 2), np.float32) -100
        batch, batchidx, inputs, lastfire, lastframet = batch_concat(batch, batchidx, lastfire, lastframet, tstamp, event)
        # print(tstamp[-1]-tstamp[0], len(batch))
        if len(inputs) > 0:
            img = np.zeros((180, 240, 3))
            img[..., (0, 2)] = inputs[-1][-1]
            cv.imshow('stream', img)
            k = cv.waitKey(1)
            if k == 'q':
                break
        with torch.no_grad():
            for inp in inputs:
                # print(inp)
                lnesint = np.sum(np.mean(inp, 0))
                if lnesint > 300 or pred is None:
                    inp = torch.tensor(np.asarray(inp), dtype=torch.float16).cuda()
                    pred = model(inp, None).detach().cpu().numpy()
                for l in pred:
                    my_filter_slow.predict()
                    my_filter_slow.update(l)
                    if np.max(np.abs(my_filter_slow.y)) > 0.7:
                        # print('fast')
                        fastcntr += 1
                        my_filter.Q = fast_Q
                        my_filter.R = fast_R
                    else:
                        my_filter.Q = slow_Q
                        my_filter.R = slow_R
                    my_filter.predict()
                    my_filter.update(l)
                    filtered = my_filter.x[::2]
                    fifowrite.write(filtered.astype(np.float64).tobytes())
                fifowrite.flush()

if __name__ == "__main__":
    main()
