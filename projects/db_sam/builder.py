
import torch
from collections.abc import Mapping
from omegaconf import DictConfig, OmegaConf

from src.models.segmentation_models import AdapterSamFactory


def build_adapted_sam_vit_b(sam_ckpt_path, model_cfg: DictConfig):
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    with open(sam_ckpt_path, "rb") as f:
        sam_state_dict = torch.load(f)
    pretrained_sam_encoder_weights = {
        k.replace("image_encoder.", ""): v for k, v in sam_state_dict.items() if 'image_encoder' in k
    }
    prompt_encoder_pretrained_weights = {
        k.replace("prompt_encoder.", ""): v for k, v in sam_state_dict.items() if 'prompt_encoder' in k
    }
    pretrained_sam_decoder_weights = {
        k.replace("mask_decoder.", ""): v for k, v in sam_state_dict.items() if 'mask_decoder' in k
    }
    pass_pretrained_sam_weights_into_model_cfg("pretrained_sam_encoder_weights", pretrained_sam_encoder_weights, model_cfg)
    pass_pretrained_sam_weights_into_model_cfg("pretrained_sam_decoder_weights", pretrained_sam_decoder_weights, model_cfg)

    sam = AdapterSamFactory().get(conf=model_cfg)
    sam.prompt_encoder.load_state_dict(prompt_encoder_pretrained_weights)
    sam.eval()

    return sam


def pass_pretrained_sam_weights_into_model_cfg(name, weights, cfg: Mapping):
    for k, v in cfg.items():
        if isinstance(v, Mapping):
            pass_pretrained_sam_weights_into_model_cfg(name, weights, v)
        else:
            if name in cfg:
                cfg[name] = weights


