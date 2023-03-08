import torch
import torch.nn as nn
import pytorch_lightning as pl
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model
from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler, EMAOptimizer
from audio_encoders_pytorch import MelE1d, TanhBottleneck
import wandb

class GuidedAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        encoder = MelE1d( # The encoder used, in this case a mel-spectrogram encoder
                in_channels=2,  # Audio channels
                channels=512,  # encoder channels
                multipliers=[1, 1],
                factors=[2],
                num_blocks=[12],
                out_channels=32,  # channels of latent representation
                mel_channels=80,  # Mel-spectrogram channels # should be 128
                mel_sample_rate=16000,
                mel_normalize_log=True,
                bottleneck=TanhBottleneck(),)
        self.model = DiffusionAE(encoder=encoder,
                inject_depth=6,
                net_t=UNetV0, # The model type used for diffusion upsampling
                in_channels=2, # U-Net: number of input/output (audio) channels
                channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
                factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], #  # U-Net: downsampling and upsampling factors at each layer
                items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
                diffusion_t=VDiffusion, # The diffusion method used
                sampler_t=VSampler,) # The diffusion sampler used)

        self.save_hyperparameters()

    def forward(self) -> torch.Tensor:
        return self.model()

    @torch.no_grad()
    def sample(self, input_array, num_steps=100) -> torch.Tensor:
        # if dim = 2, unsqueeze to dim = 3
        if input_array.dim() == 2:
            input_array = input_array.unsqueeze(0)
        # bug to fix: length bust be multiple of 2
        input_array = input_array[:, :, :65536]
        latent = self.model.encode(input_array)
        sample = self.model.decode(latent, num_steps=num_steps)
        # log wave files
        for i in range(sample.shape[0]):
            output_sample = sample[i].cpu().permute(1, 0).numpy()
            input_sample = input_array[i].cpu().permute(1, 0).numpy()
            self.logger.experiment.log({"origin/sampled audio": [wandb.Audio(input_sample, sample_rate=16000),
                                        wandb.Audio(output_sample, sample_rate=16000)]})
        return sample

    def training_step(self, batch, batch_idx):
        batch = batch.permute(0, 2, 1)
        loss = self.model(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.permute(0, 2, 1)
        loss = self.model(batch)
        self.log("val_loss", loss)
        # for each validation, only log 4 samples
        if batch_idx == 0:
            self.sample(batch, num_steps=100)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.95, 0.999),
                eps=1e-6,
                weight_decay=1e-3)
        return optimizer
