#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, models
import pytorch_lightning as pl
import gc


class BaseModel(pl.LightningModule):
    def forward(self, x, prevpos):
        raise NotImplementedError()

    def training_step(self, batch, batch_nb):
        x, prevpos, y = batch
        pred = self(x, prevpos)
        # loss = F.mse_loss(pred, y)

        mano_loss = F.mse_loss(pred[:,:6], y[:,:6])
        pos_loss = F.mse_loss(pred[:,6:9], y[:,6:9])
        z_loss = F.mse_loss(pred[:,8:9], y[:,8:9])
        rot_loss = F.mse_loss(pred[:,9:], y[:,9:])

        # position is in meters. thus 1cm rmse model would get 0.0001
        # position loss, compared to other losses that are around 0.1
        # so, let's weigh it 100*100 more, so it is comparable
        # pos_loss = pos_loss * (100*100)

        # loss = (mano_loss*6+pos_loss*3+rot_loss*3)/12
        # renormalize the loss function so the components have roughly
        # the same scale
        loss = (mano_loss*6/0.1+pos_loss*3/0.0001+rot_loss*3/0.05)/12

        tensorboard_logs = {
                'train_loss': loss,
                'train_mano_loss': mano_loss,
                'train_pos_loss': pos_loss,
                'train_z_loss': z_loss,
                'train_rot_loss': rot_loss}

        return {'loss': loss.log10(), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, prevpos, y = batch
        pred = self(x, prevpos)
        # loss = F.mse_loss(pred, y)

        mano_loss = F.mse_loss(pred[:,:6], y[:,:6])
        pos_loss = F.mse_loss(pred[:,6:9], y[:,6:9])
        z_loss = F.mse_loss(pred[:,8:9], y[:,8:9])
        rot_loss = F.mse_loss(pred[:,9:], y[:,9:])

        # pos_loss = pos_loss * (100*100)

        # loss = (mano_loss*6+pos_loss*3+rot_loss*3)/12
        loss = (mano_loss*6/0.1+pos_loss*3/0.0001+rot_loss*3/0.05)/12
        # loss = (mano_loss*6/0.04+pos_loss*3/0.00002+rot_loss*3/0.05)/12

        return {'val_loss': loss,
                'val_pos_loss': pos_loss,
                'val_mano_loss': mano_loss,
                'val_z_loss': z_loss,
                'val_rot_loss': rot_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mano_loss = torch.stack([x['val_mano_loss'] for x in outputs]).mean()
        avg_pos_loss = torch.stack([x['val_pos_loss'] for x in outputs]).mean()
        avg_z_loss = torch.stack([x['val_z_loss'] for x in outputs]).mean()
        avg_rot_loss = torch.stack([x['val_rot_loss'] for x in outputs]).mean()

        tensorboard_logs = {
                'val_loss': avg_loss,
                'val_mano_loss': avg_mano_loss,
                'val_pos_loss': avg_pos_loss,
                'val_z_loss': avg_z_loss,
                'val_rot_loss': avg_rot_loss}

        gc.collect()

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': {'val_loss': avg_loss}}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


class ReturnPrevposModel(BaseModel):
    def __init__(self):
        super(ReturnPrevposModel, self).__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, prevpos):
        return prevpos+0*self.dummy

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class MNISTModel(BaseModel):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.rn = models.resnet18(num_classes=12)

    def forward(self, x, prevpos):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.rn(x)

        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
