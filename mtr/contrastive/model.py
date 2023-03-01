import torch
import torchaudio
import math
import numpy as np
from torch import Tensor, nn
from mtr.modules.head import CLIPHead
from mtr.modules.pad import repeat_padding

class ContrastiveModel(nn.Module):
    def __init__(self, audio_encoder, text_encoder, text_type, audio_dim, text_dim, mlp_dim, temperature):
        super(ContrastiveModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.text_type = text_type
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad=True)
        self.head = CLIPHead(logit_scale=self.logit_scale)
        self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
        self.text_projector =  nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, mlp_dim, bias=False))
        self.audio_encoder.train()
        # self.text_encoder.train()
        self.a_latent = nn.Identity()
        self.t_latent = nn.Identity()

    def forward(self, audio, text, text_mask=None):
        h_audio = self.encode_audio(audio)
        h_text = self.encode_bert_text(text, text_mask)
        audio_loss = self.head(h_audio, h_text)
        text_loss = self.head(h_text, h_audio)
        loss = (audio_loss + text_loss) / 2

        audio_acc = self.head.acc(h_audio, h_text)
        text_acc = self.head.acc(h_text, h_audio)
        return loss, audio_acc, text_acc, self.logit_scale

        
    def encode_audio(self, audio):
        # check audio shape
        if audio.size(-1) == 128 and audio.size(-2) != 128:  # which means audio in (B, length, dim)
            audio = audio.transpose(1, 2)
        if audio.size(-1) < 992:
            audio = repeat_padding(audio, 992)
        if audio.size(-1) > 992:
            audio = audio[:, :, :992]
        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        z_audio = self.audio_projector(h_audio)
        return z_audio


    def encode_bert_text(self, text, text_mask=None):
        text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        z_text = self.text_projector(h_text)
        return z_text