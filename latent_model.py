import torch
import torch.nn as nn
import pytorch_lightning as pl
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model
import pickle
from ae import GuidedAE
import wandb

class GuidanceModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DiffusionModel(
                    net_t=UNetV0, # The model type used for diffusion
                    in_channels=32, # U-Net: number of input/output (audio) channels
                    channels=[64, 128, 256, 512, 512], # U-Net: channels at each layer
                    factors=[1, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
                    items=[1, 2, 2, 2, 2], # U-Net: number of repeating items at each layer
                    attentions=[0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
                    attention_heads=4, # U-Net: number of attention heads per attention block
                    attention_features=128, # U-Net: number of attention features per attention block,
                    diffusion_t=VDiffusion, # The diffusion method used
                    sampler_t=VSampler, # The diffusion sampler used
                    embedding_features=128, # U-Net: embedding features
                    use_embedding_cfg=True,
                    embedding_max_length=1,  # U-Net: text embedding maximum length (default for T5-base)
                    cross_attentions=[1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
                )
        pretrained_model_ckpt = "/import/c4dm-04/yz007/best.pth"
        # self.condition_model, self.tokenizer, self.condition_model_config = get_model(ckpt=pretrained_model_ckpt)
        # for param in self.condition_model.parameters():
        #     param.requires_grad = False
        self.vocoder = GuidedAE().load_from_checkpoint("/import/c4dm-04/yz007/checkpoints/ae-epoch=49-stable.ckpt")
        # freeze
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.latent_length = 256

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def sample(self, num_steps=100, *args, **kwargs) -> torch.Tensor:
        text_input_vecs = pickle.load(open("/homes/yz007/audio-diffusion-pytorch/text_embs.pkl", "rb")) # [4, 128]
        text_input_vecs = text_input_vecs.unsqueeze(1).to(self.device)
        noise = torch.randn(text_input_vecs.size(0), 32, self.latent_length, device=self.device)
        latent = self.model.sample(noise.to(self.device),
                                 embedding=text_input_vecs,
                                 embedding_scale=5.0,
                                 num_steps=num_steps, *args, **kwargs)
        sample = self.vocoder.model.decode(latent, num_steps=num_steps)
        for i in range(sample.shape[0]):
            output_sample = [wandb.Audio(sample[i].cpu().permute(1, 0).numpy(), sample_rate=16000)
                             for i in range(sample.shape[0])]
            self.logger.experiment.log({"sampled audio": output_sample})

    def training_step(self, batch, batch_idx):
        audio_wave, audio_condition = batch
        audio_wave = audio_wave.permute(0, 2, 1)
        latent = self.vocoder.model.encode(audio_wave)
        audio_condition = audio_condition.unsqueeze(1)
        loss = self.model(latent,
                          embedding=audio_condition,
                          embedding_mask_proba=0.1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_wave, audio_condition = batch
        audio_wave = audio_wave.permute(0, 2, 1)
        latent = self.vocoder.model.encode(audio_wave)
        audio_condition = audio_condition.unsqueeze(1)
        loss = self.model(latent,
                          embedding=audio_condition,
                          embedding_mask_proba=0.1)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.sample(num_steps=100)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=1e-3)
        return optimizer