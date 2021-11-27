"""
Info about lightning module:
https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#minimal-example
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import pathlib
import dataset

LOSSES = []


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=(1,))

    def forward(self, x):
        return self.conv(x)


class BaseLineZero(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, dct):
        dct["pressure_pred"] = 0 * dct["u_in"]

    def validation_step(self, batch, batch_idx):
        self.forward(batch)
        pred = batch["pressure_pred"]
        true = batch["pressure"].float()
        loss = F.mse_loss(pred, true)
        LOSSES.append(loss)

        # logs to tensorboard by default
        self.log_dict({"val_loss": loss})


class Net(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, dct):
        x = dct["u_in"]
        n, h = x.shape
        y = self.model(x.view(n, 1, h).float())
        dct["pressure_pred"] = y.view(n, h)
        # no need to return, since changed in place

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        self.forward(batch)
        pred = batch["pressure_pred"]
        true = batch["pressure"].float()
        loss = F.mse_loss(pred, true)

        # logs to tensorboard by default
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.forward(batch)

        pred = batch["pressure_pred"]
        true = batch["pressure"].float()
        loss = F.mse_loss(pred, true)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    net = Net(model=LinearModel())

    # dummy data
    # dct = {"u_in": np.random.rand(10)}
    # print(net(dct))

    # actual data
    ds = dataset.Dataset("data/train_subset.csv")
    loader = torch.utils.data.DataLoader(ds)
    for i, item in enumerate(loader):
        if i != 0:
            continue
        net(item)
        print(item)
        item.keys()
