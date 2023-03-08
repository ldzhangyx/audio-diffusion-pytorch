import torch
import torch.nn as nn
import pytorch_lightning as pl
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model
import pickle
from ae import GuidedAE
import wandb
from mtr.modules.audio_rep import TFRep
import numpy as np


class GuidanceModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DiffusionModel(
            net_t=UNetV0,  # The model type used for diffusion
            in_channels=32,  # U-Net: number of input/output (audio) channels
            channels=[128, 128, 256, 512, 512, 1024],  # U-Net: channels at each layer
            factors=[1, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
            items=[1, 2, 2, 2, 2, 2],  # U-Net: number of repeating items at each layer
            attentions=[0, 1, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
            attention_heads=8,  # U-Net: number of attention heads per attention block
            attention_features=256,  # U-Net: number of attention features per attention block,
            diffusion_t=VDiffusion,  # The diffusion method used
            sampler_t=VSampler,  # The diffusion sampler used
            embedding_features=128,  # U-Net: embedding features
            use_embedding_cfg=True,
            embedding_max_length=1,  # U-Net: text embedding maximum length (default for T5-base)
            cross_attentions=[1, 1, 1, 1, 1, 1],  # U-Net: cross-attention enabled/disabled at each layer
        )
        pretrained_model_ckpt = "/import/c4dm-04/yz007/best.pth"
        self.condition_model, self.tokenizer, self.condition_model_config = get_model(ckpt=pretrained_model_ckpt)
        for param in self.condition_model.parameters():
            param.requires_grad = False
        self.vocoder = GuidedAE().load_from_checkpoint(
            "/import/c4dm-04/yz007/checkpoints/ae-epoch=79-val_loss=0.0720.ckpt")
        # freeze
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.latent_length = 256
        self.rep = TFRep(
            sample_rate=16000,
            f_min=0,
            f_max=8000,
            n_fft=1024,
            win_length=1024,
            hop_length=int(0.01 * 16000),
            n_mels=128,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def sample(self, num_steps=400, *args, **kwargs) -> torch.Tensor:
        text_input_vecs = pickle.load(open("/homes/yz007/audio-diffusion-pytorch/text_embs_new.pkl", "rb"))  # [4, 128]
        text_input_vecs = text_input_vecs.unsqueeze(1).to(self.device)
        noise = torch.randn(text_input_vecs.size(0), 32, self.latent_length, device=self.device)
        latent = self.model.sample(noise.to(self.device),
                                   embedding=text_input_vecs,
                                   embedding_scale=10.0,
                                   num_steps=num_steps, *args, **kwargs)
        sample = self.vocoder.model.decode(latent, num_steps=num_steps)
        for i in range(sample.shape[0]):
            output_sample = [wandb.Audio(sample[i].cpu().permute(1, 0).numpy(), sample_rate=16000)
                             for i in range(sample.shape[0])]
            self.logger.experiment.log({"sampled audio": output_sample})

    def training_step(self, audio_wave, batch_idx):
        audio_wave = audio_wave.permute(0, 2, 1)
        audio_wave_mixed, _ = self.in_batch_mixup(audio_wave)
        audio_condition = self.get_audio_condition(audio_wave_mixed)
        latent = self.vocoder.model.encode(audio_wave_mixed)
        audio_condition = audio_condition.unsqueeze(1)
        loss = self.model(latent,
                          embedding=audio_condition,
                          embedding_mask_proba=0.1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, audio_wave, batch_idx):
        audio_condition_origin = self.get_audio_condition(audio_wave)
        audio_wave = audio_wave.permute(0, 2, 1)
        audio_wave_mixed, _ = self.in_batch_mixup(audio_wave)
        audio_condition = self.get_audio_condition(audio_wave_mixed)
        latent = self.vocoder.model.encode(audio_wave_mixed)
        audio_condition = audio_condition.unsqueeze(1)
        loss = self.model(latent,
                          embedding=audio_condition,
                          embedding_mask_proba=0.1)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.sample(num_steps=400)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=1e-3)
        return optimizer

    @torch.no_grad()
    def get_audio_condition(self, batch):
        # if batch.size(1) != 2:
        #     batch = batch.permute(0, 2, 1).mean(dim=1)  # batch => (batch_size, segment_length)
        if len(batch.size()) == 3:
            batch = batch.mean(dim=1)
        spec = self.rep.melspec(batch)
        audio_embedding = self.condition_model.encode_audio(spec)
        return audio_embedding

    def in_batch_mixup(self, batch, labels=None, mix_possibility=0.5, mix_beta=5):
        batch_size = batch.size(0)  # batch: (bs, channel, segment_length)
        is_mix = torch.from_numpy(np.random.choice(np.array([0, 1]),
                                                   size=batch_size, p=[1 - mix_possibility, mix_possibility])).to(self.device)
        mix_lambda = torch.from_numpy(np.random.beta(mix_beta, mix_beta, size=batch_size)).to(self.device)
        mix_lambda = torch.clamp(mix_lambda, 0.1, 0.9)
        permuted_batch_index = torch.randperm(batch_size).to(self.device)
        # if is_mix[i] == 1, then use mix_lambda[i] * batch[i] + (1 - mix_lambda[i]) * batch[permuter_batch_index[i]]
        # else use batch[i]
        batch = batch * is_mix.view(-1, 1, 1).float() + (mix_lambda.view(-1, 1, 1).float() * batch +
                 (1 - mix_lambda.view(-1, 1, 1).float()) * batch[permuted_batch_index]) * (1 - is_mix).view(-1, 1, 1).float()
        if labels is not None:
            labels = labels * is_mix.view(-1, 1).float() + (mix_lambda.view(-1, 1).float() * labels +
                      (1 - mix_lambda.view(-1, 1).float()) * labels[permuted_batch_index]) * (1 - is_mix).view(-1, 1).float()
        return batch, labels



