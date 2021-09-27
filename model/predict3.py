#!/usr/bin/env python3
import numpy as np
import scipy.ndimage.morphology as morph
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms, models
import pytorch_lightning as pl
import os
import sys
from datetime import datetime
import struct
from tqdm import tqdm
from dv import AedatFile
import cv2 as cv

from fastevc import EVCDatasetFast
from model import ReturnPrevposModel, MNISTModel


# DATASETS = ['new', 'new2']
DATASETS = ['new_handfingersvlad_hard_ng', 'new_handfingersvlad_hard_ng2']
WIDTH, HEIGHT = 240, 180

def undistort(xy):
    """ Undistort the points using computed distortion matrices of our DAVIS240C """
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
    """ Read and splice all the events from file 'name' into windows of 1/fps ms each """

    # try to load from cache first instead of resplicing the events from the aedat4
    cachefn = os.path.join('aedatcache', os.path.basename(name)+'.cache'+str(fps))
    if os.path.exists(cachefn):
        print('loaded events from cache:', cachefn)
        return np.load(cachefn, allow_pickle=True)['events']
    if os.path.exists(cachefn+'.npz'):
        print('loaded events from cache:', cachefn+'.npz')
        return np.load(cachefn+'.npz', allow_pickle=True)['events']

    with AedatFile(name) as f:
        start = None
        prevts = 0
        events = []
        for e in f['events']:
            if start is None:
                start = e.timestamp
                prevts = e.timestamp
            ts = e.timestamp-start

            # compute the window idx
            idx = ts*fps//1000000
            # create enough empty windows in the array to accomodate the current event
            while len(events) <= idx:
                events.append([])

            # add the event
            events[idx].append((e.x, e.y, 0 if e.polarity else 1))

            # if ts/1000000 > 10:
            #     break
            # print the loading progress every 1s of loaded time
            if (e.timestamp-start) % 1000000 < (prevts-start) % 1000000:
                print((e.timestamp-start)/1000000)
            prevts = e.timestamp

        # undistort all the events
        for i in range(len(events)):
            events[i] = np.array(events[i], np.float32)
            if len(events[i]) > 0:
                # split into coordinates and polarity
                xy, p = events[i][:, 0:2], events[i][:, 2]
                xy = undistort(xy.astype(np.float32))

                # recombine them back
                events[i] = np.c_[xy, p]
                # print(xy.shape, events[i].shape)
            events[i] = np.array(events[i], np.int16)

        # save the cached events so they could be reloaded later without recomputing all the stuff
        np.savez_compressed(cachefn, events=events)
        print('saved events to cache:', cachefn)
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

    # def __getitem__(self, idx):
    #     img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
    #     for i in range(self.WINDOW):
    #         if len(self.events[idx+i]) > 0:
    #             xs, ys, ps = self.events[idx+i].T
    #             img[ys, xs, ps] = i/self.WINDOW
    #         # for x, y, p in self.events[idx+i]:
    #         #     img[y, x, p] = i/self.WINDOW
    #     X = img
    #     return torch.tensor(X, dtype=torch.float32)

    def __getitem__(self, idx):
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        for i in range(self.WINDOW):
            if len(self.events[idx+i]) > 0:
                xs, ys, ps = self.events[idx+i].T
                img[ys, xs, ps] = i/self.WINDOW
            # for x, y, p in self.events[idx+i]:
            #     img[y, x, p] = i/self.WINDOW
        X = img

        # compute the mask used for filtering noise events
        # it is a minimum of vertical and horizontal grayscale morphological openings of the lnes
        # where the mask is 0, we should filter these event out
        av = morph.grey_opening(img, size=(2, 1, 1))
        ah = morph.grey_opening(img, size=(1, 2, 1))
        a = np.minimum(av, ah)

        # compute LNES
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        for i in range(self.WINDOW):
            if len(self.events[idx+i]) > 0:
                xs, ys, ps = self.events[idx+i].T
                # detect and fix flickering that happens when we shoot both the grayscale frames and events on DAVIS240C
                cur = (self.events[idx+i][:, 2]==0).sum()
                prv = (self.events[idx+i-1][:, 2]==0).sum() if len(self.events[idx+i-1]) > 0 else 0
                nxt = (self.events[idx+i+1][:, 2]==0).sum() if len(self.events[idx+i+1]) > 0 else 0

                # if the number of events is larger in this frame is larger than in the both consecutive ones, fix it
                if prv < cur or nxt < cur:
                    # res = img.copy()
                    # res[ys, xs, ps] = i/self.WINDOW
                    # resv = morph.grey_opening(res, (2, 1, 1))
                    # resh = morph.grey_opening(res, (1, 2, 1))
                    # res = np.minimum(resv, resh)

                    # use the mask we computed earlier to fix the flickering noise events out
                    mask = a[ys, xs, ps] > 0
                    # mask = ((av[ys, xs, ps] > 0) | (ah[ys, xs, ps] > 0))
                    ys, xs, ps = ys[mask], xs[mask], ps[mask]
                img[ys, xs, ps] = i/self.WINDOW
            # for x, y, p in self.events[idx+i]:
            #     img[y, x, p] = i/self.WINDOW
        X = img

        return torch.tensor(X, dtype=torch.float32)



if __name__ == '__main__':
    testconcat = AedatDataset()
    testconcat.load(sys.argv[2])
    test_loader = DataLoader(testconcat, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = MNISTModel.load_from_checkpoint(
        checkpoint_path=sys.argv[1]
    )
    model = model.cuda()
    model.eval()
    with open(sys.argv[3], 'w') as prf:
        prevposes = None
        # load all the batches, leave out the last 500 ms of windows
        for x, bidx in zip(test_loader, tqdm(range((len(testconcat)-500)//test_loader.batch_size))):
            # bidx is the current batch number, so the start is WINDOW ms before that
            # startidx is when the first element of the batch starts
            startidx = bidx * test_loader.batch_size - testconcat.WINDOW
            # endidx = startidx + testconcat.WINDOW
            # endidx is when the last element of the batch starts
            endidx = startidx + test_loader.batch_size

            if startidx < 0:
                # if there's no source for the prevpos, set it to zeros
                prevpos = torch.zeros((x.size(0), 12))
            else:
                # else, use prevposes
                prevpos = torch.tensor(prevposes[startidx:endidx])

            # predict and transfer the result to cpu
            pred = model(x.cuda(), prevpos.cuda())
            pred = pred.clone().detach().cpu().numpy()

            # set or concatenate to previous prevposes
            if prevposes is None:
                prevposes = pred
            else:
                prevposes = np.r_[prevposes, pred]

            # output the results
            for l in pred:
                print(*l, file=prf)

