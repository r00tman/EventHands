#!/usr/bin/env python3
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
import time
from tqdm import tqdm
from fastevc import EVCDatasetFast
from model import ReturnPrevposModel, MNISTModel


#DATASETS = ['new', 'new2']
# DATASETS = ['new']
# DATASETS = ['new2', 'new_handfingersvlad2']
# DATASETS = ['new_handfingersvlad_hard_ng', 'new_handfingersvlad_hard_ng2', 'new_handfingersvlad_hard_translations', 'new_handfingersvlad_hard_translations2']
# oom below w/40k
# DATASETS = ['new_handfingersvlad_hard_ng', 'new_handfingersvlad_hard_ng2', 'new_handfingersvlad_hard_translations_bothhandsides', 'new_handfingersvlad_hard_translations_bothhandsides1',
#             'new_handfingersvlad_hard_translations_bottom','new_handfingersvlad_hard_translations_bottom2' ]
# DATASETS = ['new_handfingersvlad_hard_translations_bothhandsides', 'new_handfingersvlad_hard_translations_bothhandsides1',
#             'new_handfingersvlad_hard_translations_bottom','new_handfingersvlad_hard_translations_bottom2' ]
DATASETS = ['new_handfingersvlad_hard_ng', 'new_handfingersvlad_hard_ng2', 'new_handfingersvlad_hard_translations_bothhandsides', 'new_handfingersvlad_hard_translations_bothhandsides1']
# DATA_DIR = 'data'
DATA_DIR = '/scratch/inf0/user/vrudnev/data'


if __name__ == '__main__':
    trainconcat = []
    testconcat = []
    for name in DATASETS:
        start = time.time()
        ds = EVCDatasetFast()
        print('loading dataset', name)
        ds.load(os.path.join(DATA_DIR, name), 0, 1000*40200)
        print('loaded dataset', name, 'in', time.time()-start, 'seconds')
        traindataset = ds.view(1000*250, 1000*(40000-250))
        print('loaded train dataset')
        testdataset = ds.view(0, 1000*250)
        print('loaded test dataset')
        trainconcat += [traindataset]
        testconcat += [testdataset]

    trainconcat = ConcatDataset(trainconcat)
    testconcat = ConcatDataset(testconcat)

    train_loader = DataLoader(trainconcat, batch_size=64, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(testconcat, batch_size=64, num_workers=4, pin_memory=False)

    model = MNISTModel()
    trainer = pl.Trainer(gpus=1,
                         progress_bar_refresh_rate=20,
                         max_epochs=1000,
                         val_check_interval=15000,
                         logger=pl.loggers.TensorBoardLogger(
                             save_dir='',
                             name='logs',
                             version=os.environ['DIRNAME']),
                         checkpoint_callback=ModelCheckpoint(
                             filepath=None,
                             monitor='val_loss',
                             save_last=True,
                             verbose=True,
                             mode='min',
                             period=0))
    trainer.fit(model, train_loader, test_loader)
    # print(trainer.test(model, test_loader))
