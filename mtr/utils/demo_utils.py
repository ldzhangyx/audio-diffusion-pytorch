
import os
import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, set_seed
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.modules.head import ClsHead
from mtr.contrastive.model import ContrastiveModel


def get_model(framework="contrastive", ckpt=None):
    save_dir = f"mtr/{framework}/"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    audio_preprocessr = TFRep(
                sample_rate= 16000,
                f_min=0,
                f_max= 8000,
                n_fft = config.n_fft,
                win_length = config.win_length,
                hop_length = int(0.01 * config.sr),
                n_mels = config.mel_dim
    )
    frontend = ResFrontEnd(
        input_size=(config.mel_dim, int(100 * config.duration) + 1), # 128 * 992
        conv_ndim=128, 
        attention_ndim=config.attention_ndim,
        mix_type= config.mix_type
    )
    audio_encoder = MusicTransformer(
        audio_representation=audio_preprocessr,
        frontend = frontend,
        audio_rep = config.audio_rep,
        attention_nlayers= config.attention_nlayers,
        attention_ndim= config.attention_ndim
    )
    text_encoder = AutoModel.from_pretrained(config.text_backbone)
    tokenizer = AutoTokenizer.from_pretrained(config.text_backbone)
    config.text_dim = 768
    config.audio_dim = config.attention_ndim
    model = ContrastiveModel(
        audio_encoder= audio_encoder,
        text_encoder= text_encoder,
        text_type = config.text_type,
        audio_dim= config.audio_dim,
        text_dim= config.text_dim,
        mlp_dim= config.mlp_dim,
        temperature = config.temperature
    )
    pretrained_object = torch.load(ckpt, map_location=torch.device('cuda'))
    state_dict = pretrained_object['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]              
    model.load_state_dict(state_dict, strict=False)
    return model, tokenizer, config