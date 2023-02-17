import pytorch_lightning as pl
import torch
from model import GuidanceModel
from data import Music4AllDataset, Music4AllDataModule
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
BATCH_SIZE = 32

def main():
    pl.seed_everything(42)
    guidance_model = GuidanceModel()
    data_module = Music4AllDataModule(batch_size=BATCH_SIZE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="/import/c4dm-04/yz007/checkpoints",
        filename="mousai-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(
        enable_checkpointing=True,
        # default_root_dir="/import/c4dm-04/yz007/checkpoints",
        accelerator='gpu',
        # auto_select_gpus=True,
        # strategy='ddp',
        # devices=1,
        precision=16,
        log_every_n_steps=1,
        logger=pl.loggers.TensorBoardLogger("logs", name="guidance"),
        max_epochs=50,
        val_check_interval=0.1,
        limit_val_batches=0.5,
        callbacks=[checkpoint_callback]  # , FinetuningScheduler()],
    )
    trainer.fit(guidance_model,
                datamodule=data_module,
                )


if __name__ == "__main__":
    main()
