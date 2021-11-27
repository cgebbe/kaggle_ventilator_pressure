import pytorch_lightning as pl

pl.seed_everything(42)

import torch.utils.data
import modules
import models.unet_1d
import dataset
import numpy as np


if __name__ == "__main__":
    loader_kwargs = dict(
        num_workers=4,
        pin_memory=True,
        batch_size=10,
        drop_last=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset.Dataset("data/train_5000.csv"),
        **loader_kwargs,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset.Dataset("data/valid_2000.csv"), **loader_kwargs
    )

    trainer = pl.Trainer(
        # hyperparams
        max_epochs=20,
        # hardware
        accelerator="gpu",
        devices=1,
        deterministic=True,
    )

    if 0:
        result = trainer.validate(modules.BaseLineZero(), val_loader, verbose=True)
        # list corresponds to length of passed dataloaders
        assert False  # val_loss = 191.7105
        np.testing.assert_array_almost_equal(
            torch.mean(torch.stack(models.LOSSES)).cpu().numpy(),
            result[0]["val_loss"],
            decimal=3,
        )  # is the same as manual computation

    if 1:
        model = modules.Net(
            # model=modules.LinearModel(),
            model=models.unet_1d.UNet(n_channels=1, n_classes=1, bilinear=False),
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        assert False
