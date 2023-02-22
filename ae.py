import torch
import torch.nn as nn
import pytorch_lightning as pl
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model
from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler, EMAOptimizer
from audio_encoders_pytorch import MelE1d, TanhBottleneck

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
                mel_channels=80,  # Mel-spectrogram channels
                mel_sample_rate=16000,
                mel_normalize_log=True,
                bottleneck=TanhBottleneck(),)
        self.model = DiffusionAE(encoder=encoder,
                inject_depth=6,
                net_t=UNetV0, # The model type used for diffusion upsampling
                in_channels=2, # U-Net: number of input/output (audio) channels
                channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
                factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
                items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
                diffusion_t=VDiffusion, # The diffusion method used
                sampler_t=VSampler,) # The diffusion sampler used)

    def forward(self) -> torch.Tensor:
        return self.model()

    @torch.no_grad()
    def sample(self, text = "piano", num_steps=100, length=81920) -> torch.Tensor:
        noise = torch.randn(1, 2, length)
        latent = self.model.encode(noise)
        sample = self.model.decode(latent, num_steps=100)
        # write wav file

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
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.95, 0.999),
                eps=1e-6,
                weight_decay=1e-3),
        return optimizer
