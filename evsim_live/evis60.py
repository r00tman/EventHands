#!/usr/bin/env python3
import numpy as np
import scipy.ndimage.morphology as morph
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dv import AedatFile
import cv2 as cv
import ffmpeg
import sys

WIDTH, HEIGHT = 240, 180

def undistort(xy):
    return xy
    mtx = np.array(
            [[252.91294004, 0, 129.63181808],
            [0, 253.08270535, 89.72598511],
            [0, 0, 1.]])
    dist = np.array(
            [-3.30783118e+01,  3.40196626e+02, -3.19491618e-04, -6.28058571e-04,
            1.67319020e+02, -3.27436981e+01,  3.29048638e+02,  2.85123812e+02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00])
    und = cv.undistortPoints(xy, mtx, dist)
    und = und.reshape(-1, 2)
    und = np.c_[und, np.ones_like(und[:,0])] @ mtx.T
    assert(np.allclose(und[:, 2], np.ones_like(und[:, 2])))
    und = und[:, :2]

    und[:, 0] = np.clip(und[:, 0], 0, WIDTH-1)
    und[:, 1] = np.clip(und[:, 1], 0, HEIGHT-1)
    assert(np.all(0 <= und) and np.all(und[:, 0] < WIDTH) and np.all(und[:, 1] < HEIGHT))
    return und

def read_event(name, fps=1000):
    with AedatFile(name) as f:
        start = None
        prevts = 0
        events = []
        for e in f['events']:
            if start is None:
                start = e.timestamp
                prevts = e.timestamp
            ts = e.timestamp-start

            idx = ts*fps//1000000
            while len(events) <= idx:
                events.append([])

            events[idx].append((e.x, e.y, 0 if e.polarity else 1))

            # if ts/1000000 > 10:
            #     break
            if (e.timestamp-start) % 1000000 < (prevts-start) % 1000000:
                print((e.timestamp-start)/1000000)
            prevts = e.timestamp

        for i in range(len(events)):
            events[i] = np.array(events[i], np.float32)
            if len(events[i]) > 0:
                xy, p = events[i][:, 0:2], events[i][:, 2]
                xy = undistort(xy.astype(np.float32))

                events[i] = np.c_[xy, p]
                # print(xy.shape, events[i].shape)
            events[i] = np.array(events[i], np.int16)


        return events

class AedatDataset(Dataset):
    WINDOW = 100

    def __init__(self):
        pass

    def load(self, name):
        self.events = read_event(name)
        print(len(self.events))

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        for i in range(self.WINDOW):
            if len(self.events[idx+i]) > 0:
                xs, ys, ps = self.events[idx+i].T
                img[ys, xs, ps] = i/self.WINDOW
            # for x, y, p in self.events[idx+i]:
            #     img[y, x, p] = i/self.WINDOW
        X = img

        av = morph.grey_opening(img, size=(2, 1, 1))
        ah = morph.grey_opening(img, size=(1, 2, 1))
        a = np.minimum(av, ah)

        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        for i in range(self.WINDOW):
            if len(self.events[idx+i]) > 0:
                xs, ys, ps = self.events[idx+i].T
                cur = (self.events[idx+i][:, 2]==0).sum()
                prv = (self.events[idx+i-1][:, 2]==0).sum() if len(self.events[idx+i-1]) > 0 else 0
                nxt = (self.events[idx+i+1][:, 2]==0).sum() if len(self.events[idx+i+1]) > 0 else 0
                if prv < cur or nxt < cur:
                    # res = img.copy()
                    # res[ys, xs, ps] = i/self.WINDOW
                    # resv = morph.grey_opening(res, (2, 1, 1))
                    # resh = morph.grey_opening(res, (1, 2, 1))
                    # res = np.minimum(resv, resh)
                    mask = a[ys, xs, ps] > 0
                    # mask = ((av[ys, xs, ps] > 0) | (ah[ys, xs, ps] > 0))
                    ys, xs, ps = ys[mask], xs[mask], ps[mask]
                img[ys, xs, ps] = i/self.WINDOW
            # for x, y, p in self.events[idx+i]:
            #     img[y, x, p] = i/self.WINDOW
        X = img

        return torch.tensor(X, dtype=torch.float32)

ds = AedatDataset()
# ds.load("/home/r00tman/Recordings/handfingers_vlad/handfingers_vlad-2020_06_26_13_35_42.aedat4")
ds.load(sys.argv[1])


out = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=60)
    # .output('out.mp4', pix_fmt='yuv444p', crf=10)
    .output(sys.argv[2], pix_fmt='yuv444p', crf=10)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

for i in range(0, 80*60):
    a = ds[i*1000//60]
    p = np.zeros((HEIGHT, WIDTH, 3))
    # av = morph.grey_opening(a, size=(2, 1, 1))
    # ah = morph.grey_opening(a, size=(1, 2, 1))
    # a = np.minimum(av, ah)
    p[..., (0, 2)] = a*255

    assert(p.shape == (HEIGHT,WIDTH,3))
    out.stdin.write(p.astype(np.uint8).tobytes())

out.stdin.close()
out.wait()
