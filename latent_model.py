import torch
import torch.nn as nn
import pytorch_lightning as pl
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model

class GuidanceModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DiffusionModel(
                    net_t=UNetV0, # The model type used for diffusion
                    in_channels=2, # U-Net: number of input/output (audio) channels
                    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
                    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
                    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
                    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
                    attention_heads=8, # U-Net: number of attention heads per attention block
                    attention_features=64, # U-Net: number of attention features per attention block,
                    diffusion_t=VDiffusion, # The diffusion method used
                    sampler_t=VSampler, # The diffusion sampler used
                    embedding_features=256, # U-Net: embedding features
                    # use_embedding_cfg=True,
                    cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
                )
        pretrained_model_ckpt = "/import/c4dm-04/yz007/best.pth"
        self.condition_model, self.tokenizer, self.condition_model_config = get_model(ckpt=pretrained_model_ckpt)
        for param in self.condition_model.parameters():
            param.requires_grad = False
        self.wave_length = 81920

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def sample(self, text = "piano", num_steps=100, *args, **kwargs) -> torch.Tensor:
        noise = torch.randn(1, 2, self.wave_length)
        text_input_vec = self.tokenizer(text, return_tensors="pt")['input_ids'].cuda()
        embedding = self.condition_model.encode_bert_text(text)
        embedding = embedding.unsqueeze(1).unsqueeze(1)
        return self.model.sample(noise,
                                 embedding=embedding,
                                 num_steps=num_steps, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        audio_wave = batch
        audio_embedding = self.condition_model.encode_audio(audio_wave)
        loss = self.model(audio_wave, embedding=audio_embedding)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_wave = batch
        audio_embedding = self.condition_model.encode_audio(audio_wave)
        loss = self.model(audio_wave, embedding=audio_embedding)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=1e-3)
        return optimizer