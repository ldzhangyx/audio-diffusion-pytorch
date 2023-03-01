import pytorch_lightning as pl
import torch
from latent_model import GuidanceModel
from ae import GuidedAE
from data import Music4AllDataModule
import os
from audio_diffusion_pytorch import EMA
import wandb

wandb.login()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
BATCH_SIZE = 32
SAMPLING_RATE = int(16000 / 1)
SEGMENT_LENGTH = int(2**17 / 1)

def main():
    pl.seed_everything(42)
    guidance_model = GuidanceModel()
    data_module = Music4AllDataModule(batch_size=BATCH_SIZE,
                                      sample_rate=SAMPLING_RATE,
                                      segment_length=SEGMENT_LENGTH,
                                      condition=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="/import/c4dm-04/yz007/checkpoints",
        filename="latent-{epoch:02d}-{val_loss:.4f}",
    )
    ema_callback = EMA(
        decay=0.995,
    )
    trainer = pl.Trainer(
        enable_checkpointing=True,
        # default_root_dir="/import/c4dm-04/yz007/checkpoints",
        accelerator='gpu',
        # auto_select_gpus=True,
        # strategy='ddp',
        # devices=2,
        precision=16,
        log_every_n_steps=1,
        logger=pl.loggers.WandbLogger(project=f"latent-{SAMPLING_RATE}",
                                      ),
        max_epochs=80,
        val_check_interval=0.2,
        limit_val_batches=300,
        limit_train_batches=8000,
        callbacks=[checkpoint_callback, ema_callback]
    )
    # guidance_model = guidance_model.load_from_checkpoint("/import/c4dm-04/yz007/checkpoints/ae-epoch=01-val_loss=0.10.ckpt")
    trainer.fit(guidance_model,
                # ckpt_path="/import/c4dm-04/yz007/checkpoints/ae-epoch=49-val_loss=0.0723.ckpt",
                datamodule=data_module,
                )


if __name__ == "__main__":
    main()