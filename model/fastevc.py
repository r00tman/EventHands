import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import struct
import gc
from tqdm import tqdm
import evcreader


class EVCDataset(Dataset):
    def __init__(self):
        pass

    def load(self, name, offset, count):
        evt = open(name+".evc", "rb")
        meta = open(name+".meta", "rb")
        ncomps, = struct.unpack("<i", meta.read(4))

        self.offset = offset
        self.count = count

        self.events = []
        self.pos = []

        # skip first frame
        # gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
        # magic = struct.unpack("<BB", meta.read(2))
        # assert(magic == [4, 13])

        # print(gt, magic)

        # ignore first new frame construct
        x, y, p = struct.unpack('<HBB', evt.read(4))
        assert(p == 255)
        # # ignore first frame
        # x, y, p = struct.unpack('<HBB', evt.read(4))
        # print(x, y, p)
        # while True:
        #     x, y, p = struct.unpack('<HBB', evt.read(4))
        #     if p == 255:
        #         break

        # start reading frames
        for idx in tqdm(range(self.offset+self.count)):
            # read events
            ev = []
            while True:
                x, y, p = struct.unpack('<HBB', evt.read(4))
                if p == 255:
                    break
                ev.append((x, y, p))
            ev = np.array(ev, np.int16)
            self.events.append(ev)

            # read metadata
            gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
            magic = struct.unpack("<BB", meta.read(2))
            if idx == 0:
                print(gt, magic)
            assert(magic == (4, 13))

            self.pos.append(np.array(gt, np.float32))
        evt.close()
        meta.close()

    def view(self, offset, count):
        assert(self.offset + offset + count <= self.count)
        res = EVCDataset()
        res.offset = self.offset + offset
        res.count = count
        res.events = self.events
        res.pos = self.pos
        return res

    def __len__(self):
        return self.count

    WINDOW = 100
    def __getitem__(self, idx):
        WIDTH, HEIGHT = 240, 180
        WINDOW = 100
        idx += self.offset
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        prevpos = self.pos[idx]
        for i in range(WINDOW):
            for x, y, p in self.events[idx+i]:
                img[y, x, p] = i/WINDOW

            # ignore if it's randomizer transition frame that
            # is every 50 seconds in the dataset
            transition_frame = ((idx+i)%(50*1000) == 0)
            if transition_frame:  # TODO: what if it's the last frame?
                img *= 0
                prevpos = self.pos[idx+i]
            pos = self.pos[idx+i]
        X = torch.tensor(img, dtype=torch.float32)
        prevpos = torch.tensor(prevpos, dtype=torch.float32)
        Y = torch.tensor(pos, dtype=torch.float32)
        return X, prevpos, Y

class EVCDatasetFast(Dataset):
    def __init__(self):
        pass

    def load(self, name, offset, count):
        # evt = open(name+".evc", "rb")
        meta = open(name+".meta", "rb")
        ncomps, = struct.unpack("<i", meta.read(4))

        self.offset = offset
        self.count = count

        # self.events = []
        self.pos = []

        # skip first frame
        # gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
        # magic = struct.unpack("<BB", meta.read(2))
        # assert(magic == [4, 13])

        # print(gt, magic)

        # # ignore first frame
        # x, y, p = struct.unpack('<HBB', evt.read(4))
        # print(x, y, p)
        # while True:
        #     x, y, p = struct.unpack('<HBB', evt.read(4))
        #     if p == 255:
        #         break

        # start reading event camera frames
        self.events = evcreader.read(name+".evc", self.offset+self.count)
        # evtdt = np.dtype([('x', np.uint16), ('y', np.uint8), ('p', np.uint8)])
        # BUF_SIZE = 65536*512
        # # BUF_SIZE = 512
        # buf = np.fromfile(evt, dtype=evtdt, count=BUF_SIZE)
        # new_frame_pos = np.where(buf['p']==255)[0]
        # nfp_off = 0
        # # buf = np.hsplit(buf, new_frame_pos)
        # # print(buf)

        # for idx in tqdm(range(self.offset+self.count)):
        #     # read events
        #     if idx+1-nfp_off >= len(new_frame_pos):
        #         tmpbuf = np.fromfile(evt, dtype=evtdt, count=BUF_SIZE)
        #         # read until we find the next frame starting point
        #         while np.all(tmpbuf['p'] != 255):
        #             tmpbuf = np.r_[tmpbuf, np.fromfile(evt, dtype=evtdt, count=BUF_SIZE)]
        #         # concat with old data
        #         buf = np.r_[buf[new_frame_pos[idx-nfp_off]:], tmpbuf]
        #         # reset the frame indexing starting with the current frame
        #         new_frame_pos = np.where(buf['p']==255)[0]
        #         nfp_off = idx
        #     # extract the frame, unite the columns
        #     ev = buf[new_frame_pos[idx-nfp_off]+1:new_frame_pos[idx+1-nfp_off]]
        #     ev = np.c_[ev['x'].astype(np.uint8),
        #                ev['y'].astype(np.uint8),
        #                ev['p'].astype(np.uint8)]
        #     self.events.append(ev)

        # read metadata: ncomps of doubles and the 2-byte magic
        metadt = np.dtype([('data', np.float64, ncomps), ('m0', np.uint8), ('m1', np.uint8)])
        self.pos = np.fromfile(meta, dtype=metadt, count=(self.offset+self.count))
        # check the magic
        assert(np.all(self.pos['m0'] == 4) and np.all(self.pos['m1'] == 13))
        # convert to float
        self.pos = self.pos['data'].astype(np.float32)

        # for idx in tqdm(range(self.offset+self.count)):
        #     # read metadata
        #     gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
        #     magic = struct.unpack("<BB", meta.read(2))
        #     if idx == 0:
        #         print(gt, magic)
        #     assert(magic == (4, 13))

        #     self.pos.append(np.array(gt, np.float32))
        # evt.close()
        meta.close()

    def view(self, offset, count):
        assert(self.offset + offset + count <= self.count)
        res = type(self)()
        res.offset = self.offset + offset
        res.count = count
        res.events = self.events
        res.pos = self.pos
        return res

    def __len__(self):
        return self.count

    WINDOW = 100
    def __getitem__(self, idx):
        WIDTH, HEIGHT = 240, 180
        # WINDOW = 100
        def log_uniform_int(a, b):
            l = np.random.uniform(np.log(a), np.log(b))
            return int(round(np.exp(l)))
        WINDOW = log_uniform_int(30,300)  # speed augmentation
        idx += self.offset
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        prevpos = self.pos[idx+0]
        for i in range(WINDOW):
            xs, ys, ps = self.events[idx+i].T
            img[ys, xs, ps] = i/WINDOW
            # this is EOI
            # img[ys, xs, ps] = 1.
            # this is ECI
            # img[ys, xs, ps] += 1./self.WINDOW*2/2
            # if you add the next line to ECI, this is ECI-S
            # img[ys, xs, 1-ps] += 1./self.WINDOW*2/2

            # ignore if it's randomizer transition frame that
            # is every 50 seconds in the dataset
            transition_frame = ((idx+i)%(50*1000) == 0)
            if transition_frame:
                img *= 0
                prevpos = self.pos[idx+i]
            pos = self.pos[idx+i]

        # random polarity augmentation
        if np.random.randint(0, 2) == 1:
            img[:, :, (0, 1)] = img[:, :, (1, 0)]

        # global random polarity augmentation
        whatmask = np.random.randint(0, 2, size=(HEIGHT, WIDTH))
        mask0 = np.zeros((HEIGHT, WIDTH, 2), dtype=np.bool)
        mask0[..., 0][whatmask==1] = True
        mask1 = np.zeros((HEIGHT, WIDTH, 2), dtype=np.bool)
        mask1[..., 1][whatmask==1] = True

        img[mask0], img[mask1] = img[mask1], img[mask0]

        X = torch.tensor(img, dtype=torch.float32)
        prevpos = torch.tensor(prevpos, dtype=torch.float32)

        Y = torch.tensor(pos, dtype=torch.float32)
        return X, prevpos, Y

if __name__ == '__main__':
    # Test whether it's the same as the gold one
    DATASETS = ['new_handfingersvlad_hard_ng', 'new_handfingersvlad_hard_ng2']
    # DATA_DIR = 'data'
    DATA_DIR = '/scratch/inf0/user/vrudnev/data'
    COUNT = 1000*110  # 110 seconds of 1000 FPS
    # COUNT = 1000*1000
    name = DATASETS[0]
    ds1 = EVCDatasetFast()
    print('loading dataset', name)
    ds1.load(os.path.join(DATA_DIR, name), 0, COUNT)
    print('loaded dataset', name)

    ds2 = EVCDataset()
    print('loading dataset', name)
    ds2.load(os.path.join(DATA_DIR, name), 0, COUNT)
    print('loaded dataset', name)

    def cool_assert(a, b):
        if not torch.allclose(a, b):
            idx = (a != b)
            print("IDX:", torch.where(idx))
            print("A:", a[idx])
            print("B:", b[idx])
            assert(False)

    assert(len(ds1.events) == len(ds2.events))
    try:
        for i, (x, y) in enumerate(zip(ds1.events, ds2.events)):
            assert(np.allclose(x, y))
    except:
        print(i, x, y)
        raise

    assert(len(ds1.pos) == len(ds2.pos))
    try:
        for i, (x, y) in enumerate(zip(ds1.pos, ds2.pos)):
            assert(np.allclose(x, y))
    except:
        print(i, x, y)
        raise


    DIM = max(len(ds1[0]), len(ds2[0]))
    assert(DIM == 3 and len(ds1[0]) == DIM and len(ds2[0]) == DIM)

    for d in range(DIM):
        cool_assert(ds1[0*1000][d], ds2[0*1000][d])

    for d in range(DIM):
        cool_assert(ds1[10*1000][d], ds2[10*1000][d])

    for d in range(DIM):
        cool_assert(ds1[100*1000][d], ds2[100*1000][d])

    for i in tqdm(range(0, 1000*100, 123)):
        for d in range(DIM):
            cool_assert(ds1[i][d], ds2[i][d])

    print('Congrats, all is OK; fast data loading works correctly.')
