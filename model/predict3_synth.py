#!/usr/bin/env python3
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

from fastevc import EVCDatasetFast
from model import MNISTModel

DATASET = sys.argv[1]
WIDTH, HEIGHT = 240, 180


if __name__ == '__main__':
    testconcat = EVCDatasetFast()
    LEN = 1000*1200 # 1200 seconds of data

    # load LEN milliseconds of data + 10 second tail for safety
    testconcat.load(DATASET, 0, LEN+1000*10)

    test_loader = DataLoader(testconcat, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    LOGDIR = os.path.dirname(os.path.realpath(sys.argv[0]))
    model = MNISTModel.load_from_checkpoint(
        checkpoint_path=os.path.join(LOGDIR, 'checkpoints/last.ckpt')
    )
    model = model.cuda()
    model.eval()
    dirname=os.path.basename(os.path.dirname(os.path.realpath(sys.argv[0])))
    expname = DATASET+'_'+dirname
    with open(expname+'_gt.txt', 'w') as gtf, open(expname+'_pr.txt', 'w') as prf:
        # process all LEN milliseconds in batches
        for (x, pp, y), _ in zip(test_loader, tqdm(range(LEN//test_loader.batch_size))):
            pred = model(x.cuda(), pp.cuda()).clone().detach().cpu().numpy()
            gt = y.clone().detach().cpu().numpy()
            for l in pred:
                print(*l, file=prf)
            for l in gt:
                print(*l, file=gtf)

