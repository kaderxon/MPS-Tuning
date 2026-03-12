"""
MPS-Tuning: Manifold-Preserving and Sculpting Tuning for Few-Shot VLM Adaptation
Published as a conference paper at ICLR 2026

"Preserve and Sculpt: Manifold-Aligned Fine-Tuning of Vision-Language Models
 for Few-Shot Learning", Chen et al., ICLR 2026
 https://github.com/kaderxon/MPS-Tuning
"""

import os.path as osp
import os
import time
import datetime
import copy
import math
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights,
)
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.supcon_wtext import SupConLossWithText

_tokenizer = _Tokenizer()


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset text prompt templates
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_TEMPLATES = {
    "OxfordPets":           "a photo of a {}, a type of pet.",
    "OxfordFlowers":        "a photo of a {}, a type of flower.",
    "FGVCAircraft":         "a photo of a {}, a type of aircraft.",
    "DescribableTextures":  "{} texture.",
    "EuroSAT":              "a centered satellite photo of {}.",
    "StanfordCars":         "a photo of a {}.",
    "Food101":              "a photo of {}, a type of food.",
    "SUN397":               "a photo of a {}.",
    "Caltech101":           "a photo of a {}.",
    "UCF101":               "a photo of a person doing {}.",
    "ImageNet": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetSketch": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetV2": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetA": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetR": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "Skin40":        "a photo of a {}.",
    "galaxy":        "a photo of a {}.",
    "IP102":         "a photo of a {}.",
    "NWPU_RESISC45": "a centered satellite photo of {}.",
    "RFMiD":         "a fundus image of {}.",
    "TCGA12":        "a photo of a {}.",
    "NEU_CLS":       "a photo of a hot-rolled steel plate with {}.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    """Build CLIP text-based classifier weights from GPT-generated prompts."""
    with torch.no_grad():
        clip_model = clip_model.cuda()
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t for t in gpt_prompts[classname]]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=0).cuda()
    return clip_weights


def load_clip_to_cpu(cfg):
    """Load a CLIP checkpoint onto CPU, falling back to JIT if needed."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())
    return model


def get_annealed_temperature(current_epoch, total_epochs,
                             initial_temp, final_temp, strategy='cosine'):
    """
    Cosine / linear temperature annealing for the HMS sculpting loss τ′.
    Dataset groupings and schedules are described in Appendix C.
    """
    if current_epoch >= total_epochs:
        return final_temp

    progress = current_epoch / total_epochs

    if strategy == 'linear':
        return initial_temp - progress * (initial_temp - final_temp)
    elif strategy == 'cosine':
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return final_temp + (initial_temp - final_temp) * cosine_decay
    else:
        raise ValueError(f"Unknown annealing strategy: '{strategy}'")


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction via forward hooks
# ──────────────────────────────────────────────────────────────────────────────

class FeatureMapExtractor(nn.Module):
    """Register forward hooks on named ViT blocks to capture intermediate outputs."""

    def __init__(self, model, layer_names):
        super().__init__()
        self.model       = model
        self.layer_names = layer_names
        self._features   = {name: torch.empty(0) for name in layer_names}

        for layer_id in layer_names:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self._save_hook(layer_id))

    def _save_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        image_features = self.model(x)
        features = self._features
        # Reset buffer so the next forward call starts clean
        self._features = {name: torch.empty(0) for name in self.layer_names}
        return image_features, features


# ──────────────────────────────────────────────────────────────────────────────
# CLIP ViT building blocks
# ──────────────────────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """Encode class names into CLIP text features using predefined templates."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg        = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype      = clip_model.dtype

    def forward(self):
        if "ImageNet" not in self.cfg.DATASET.NAME:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            self.clip_model = self.clip_model.cuda()
            text_features = self.clip_model.encode_text(prompts)
        else:
            # Template ensemble for ImageNet variants (Zhang et al., 2022)
            temp_list = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            text_features = []
            for temp in temp_list:
                prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
                self.clip_model = self.clip_model.cuda()
                text_features.append(self.clip_model.encode_text(prompts))
            text_features = torch.stack(text_features, dim=0).mean(dim=0)
        return text_features


class LayerNorm(nn.LayerNorm):
    """Subclass LayerNorm to handle fp16: cast to fp32 before computing, then restore."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Standard CLIP ViT Transformer block, extended with a value-stream accessor
    required for the Pseudo Forward projection (Eq. 10).
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp  = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2      = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Return (x_out, q, k, v) where v follows the value stream only."""
        y = self.ln_1(x)
        y = y.permute(1, 0, 2)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v.permute(1, 0, 2)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, q, k, v

    def forward_x(self, x: torch.Tensor):
        """Standard residual forward — output features only (used at inference)."""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_v(self, x: torch.Tensor):
        """
        Value-stream forward — propagates the value branch through the remaining
        V_proj + FFN steps as required by the Pseudo Forward path (Eq. 10).
        """
        y = self.ln_1(x)
        y = y.permute(1, 0, 2)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v.permute(1, 0, 2)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v + x
        v = v + self.mlp(self.ln_2(v))
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Pseudo Forward Container  (Sec. 3.2 / Fig. 3 / Appendix D)
# ──────────────────────────────────────────────────────────────────────────────

class PseudoForwardContainer(nn.Module):
    """
    Hosts all trainable components of the MPS-Tuning visual encoder.

    Visual encoder layer groupings (Appendix D):
      Group 1 (layers  0–3) : frozen — handled externally, untouched here.
      Group 2 (layers  4–7) : frozen backbone blocks + parallel cross_adapters
                               (zero-initialized linear layers, Appendix D).
      Group 3 (layers 8–11) : fully fine-tuned (clip_vit_layer{8..11}).

    Pseudo Forward (Eq. 10):
      Intermediate token features z'^(l) cannot be directly compared to frozen
      text embeddings because they live in a different space.  The Pseudo Forward
      bypasses attention allocation and propagates z'^(l) through the remaining
      V_proj + FFN steps to obtain zhat'^(l) in the output embedding space:

          zhat'^(l) = FFN^(L) ∘ V_Proj^(L) ∘ ... ∘ FFN^(l+1) ∘ V_Proj^(l+1) (z'^(l))
    """

    def __init__(self, ln_post, visual_proj, visual, num_layers, num_layers_final):
        super().__init__()
        self.num_layers       = num_layers
        self.num_layers_final = num_layers_final

        # Copies of the CLIP projection head (also trainable)
        self.ln_post     = copy.deepcopy(ln_post)
        self.visual_proj = copy.deepcopy(visual_proj)

        # Group 2: one zero-initialized cross-adapter per layer (Appendix D)
        # Each adapter's output is added to the corresponding frozen block output
        dtype = torch.float32
        self.cross_adapters = nn.ModuleList([
            nn.Linear(768, 768, bias=False, dtype=dtype)
            for _ in range(num_layers_final, num_layers)
        ])
        for adapter in self.cross_adapters:
            nn.init.zeros_(adapter.weight)

        # Group 3: fully fine-tuned blocks, initialized from pretrained CLIP weights
        self.clip_vit_layer8  = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer9  = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer10 = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer11 = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer8.load_state_dict(visual.transformer.resblocks[8].state_dict())
        self.clip_vit_layer9.load_state_dict(visual.transformer.resblocks[9].state_dict())
        self.clip_vit_layer10.load_state_dict(visual.transformer.resblocks[10].state_dict())
        self.clip_vit_layer11.load_state_dict(visual.transformer.resblocks[11].state_dict())

    def forward(self, feature_map_list, layer_idx,
                is_last: bool = False, is_final: bool = False):
        """
        Three operation modes selected by flags:

          is_last=True   Run group-3 blocks and collect per-layer value streams
                         for the Pseudo Forward step (Eq. 10).
          is_final=True  Project collected token features into the output
                         embedding space via ln_post → visual_proj.
          default        Apply cross_adapters[layer_idx] from group 2
                         (Appendix D).
        """
        if is_last:
            # ── Group-3 blocks + Pseudo Forward value-stream collection (Eq. 10) ──
            layer8_v_list, layer9_v_list   = [], []
            layer10_v_list, layer11_v_list = [], []
            layer11_out_list               = []

            for feat in feature_map_list:
                layer8_out,  _, _, layer8_v  = self.clip_vit_layer8(feat)

                layer9_out,  _, _, layer9_v  = self.clip_vit_layer9(layer8_out)
                layer8_v  = self.clip_vit_layer9.forward_v(layer8_v)

                layer10_out, _, _, layer10_v = self.clip_vit_layer10(layer9_out)
                layer9_v  = self.clip_vit_layer10.forward_v(layer9_v)
                layer8_v  = self.clip_vit_layer10.forward_v(layer8_v)

                layer11_out, _, _, layer11_v = self.clip_vit_layer11(layer10_out)
                layer10_v = self.clip_vit_layer11.forward_v(layer10_v)
                layer9_v  = self.clip_vit_layer11.forward_v(layer9_v)
                layer8_v  = self.clip_vit_layer11.forward_v(layer8_v)

                layer8_v_list.append(layer8_v)
                layer9_v_list.append(layer9_v)
                layer10_v_list.append(layer10_v)
                layer11_v_list.append(layer11_v)
                layer11_out_list.append(layer11_out)

            # Returns: (final output, v-streams from layers 11 → 8)
            return (layer11_out_list, layer11_v_list,
                    layer10_v_list, layer9_v_list, layer8_v_list)

        if is_final:
            # ── Project each feature map into the CLIP output embedding space ──
            for i in range(len(feature_map_list)):
                feature_map_list[i] = feature_map_list[i].permute(1, 0, 2)
                feature_map_list[i] = self.ln_post(feature_map_list[i])
                feature_map_list[i] = feature_map_list[i] @ self.visual_proj
            return feature_map_list

        # ── Group 2: apply cross_adapter at position layer_idx (Appendix D) ──
        return [self.cross_adapters[layer_idx](feat) for feat in feature_map_list]

    def forward_test(self, feature_map, layer_idx,
                     is_last: bool = False, is_final: bool = False):
        """Single-sample inference path — Pseudo Forward not needed at test time."""
        if is_last:
            # Group-3 standard forward (no value-stream tracking needed)
            x = self.clip_vit_layer8.forward_x(feature_map)
            x = self.clip_vit_layer9.forward_x(x)
            x = self.clip_vit_layer10.forward_x(x)
            x = self.clip_vit_layer11.forward_x(x)
            return x

        if is_final:
            feature_map = feature_map.permute(1, 0, 2)
            feature_map = self.ln_post(feature_map[:, 0, :])
            feature_map = feature_map @ self.visual_proj
            return feature_map

        # Group-2 cross-adapter
        return self.cross_adapters[layer_idx](feature_map)


# ──────────────────────────────────────────────────────────────────────────────
# MPS-Tuning CLIP model  (Sec. 3)
# ──────────────────────────────────────────────────────────────────────────────

class MPSTuningCLIP(nn.Module):
    """
    Full MPS-Tuning model wrapping CLIP's visual encoder.

    Only pseudo_forward (PseudoForwardContainer) is trainable; the image
    encoder backbone, text encoder, and logit scale are all frozen.

    Inference logits blend the fine-tuned and zero-shot branches (Eq. 12):
        logits = α · logits_ft + (1 − α) · logits_zs
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        # ── Pre-compute and freeze text features ──
        with torch.no_grad():
            if "ImageNet" not in cfg.DATASET.NAME:
                temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
                prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
                clip_model = clip_model.cuda()
                text_features = clip_model.encode_text(prompts)
            else:
                temp_list = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
                text_features = []
                for temp in temp_list:
                    prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                    prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
                    clip_model = clip_model.cuda()
                    text_features.append(clip_model.encode_text(prompts))
                text_features = torch.stack(text_features, dim=0).mean(dim=0)

            # Single buffer shared by both training and evaluation forward passes
            self.text_features = text_features.cuda()
            clip_model = clip_model.cpu()

        self.text_encoder  = TextEncoder(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale   = clip_model.logit_scale
        self.dtype         = clip_model.dtype

        # Layer group boundaries (Appendix D)
        self.num_layers       = 8  # total layers managed by the adapter/fine-tune path
        self.num_layers_final = 4  # layers in group 3 (fully fine-tuned)

        # Hook-based extractors for training (two augmented views required)
        hook_layers_train = [
            'transformer.resblocks.{}'.format(11 - self.num_layers),  # early-exit layer
            'transformer.resblocks.11',                                # final layer
        ]
        self.feat_extractor_v1   = FeatureMapExtractor(self.image_encoder, hook_layers_train)
        self.feat_extractor_v2   = FeatureMapExtractor(self.image_encoder, hook_layers_train)

        # Hook-based extractor for evaluation (single view, early-exit layer only)
        hook_layers_test = ['transformer.resblocks.{}'.format(11 - self.num_layers)]
        self.feat_extractor_test = FeatureMapExtractor(self.image_encoder, hook_layers_test)

        # All trainable parameters are contained here
        self.pseudo_forward = PseudoForwardContainer(
            clip_model.visual.ln_post,
            clip_model.visual.proj,
            clip_model.visual,
            num_layers=self.num_layers,
            num_layers_final=self.num_layers_final,
        )

        # α: logits interpolation weight (Eq. 12)
        self.alpha = 0.3

    def forward(self, image, image1):
        """
        Training forward pass over two augmented views.

        Returns
        -------
        logits                  Combined logits Eq. (12).
        logits_zs_list          Zero-shot logits per view.
        logits_ft_list          Fine-tuned logits per view.
        feature_ft_list         Fine-tuned all-token features per view.
        mar_local_loss          L_MAR^local  — token-level Gram alignment, Eq. (6).
        mar_global_loss         L_MAR^global — batch-level Gram alignment, Eq. (5).
        feature_ft_list_layer8  Intermediate v-stream features, layer 8  (Eq. 11).
        feature_ft_list_layer9  Intermediate v-stream features, layer 9  (Eq. 11).
        feature_ft_list_layer10 Intermediate v-stream features, layer 10 (Eq. 11).
        feature_ft_list_layer11 Layer-11 raw output features              (Eq. 11).
        feature_ft_list_layer11_v Layer-11 v-stream features              (Eq. 11).
        image_features_list     Zero-shot [CLS] features per view.
        """
        text_features = self.text_features
        early_key = 'transformer.resblocks.{}'.format(11 - self.num_layers)
        final_key  = 'transformer.resblocks.11'

        # ── Extract frozen CLIP features — no gradients ──
        with torch.no_grad():
            image_features,     feat_dict_v1 = self.feat_extractor_v1(image.type(self.dtype))
            image_features_aug, feat_dict_v2 = self.feat_extractor_v2(image1.type(self.dtype))

            # Early-exit features fed into the cross-adapter path
            feature_map_layer     = feat_dict_v1[early_key]
            feature_map_layer_aug = feat_dict_v2[early_key]

            # All-token features from the final frozen layer, used in MAR (Eq. 5/6)
            feature_map_final     = feat_dict_v1[final_key]
            feature_map_final_aug = feat_dict_v2[final_key]
            feature_map_final     = self.image_encoder.ln_post(feature_map_final.permute(1, 0, 2))
            feature_map_final_aug = self.image_encoder.ln_post(feature_map_final_aug.permute(1, 0, 2))
            feature_map_final     = feature_map_final     @ self.image_encoder.proj
            feature_map_final_aug = feature_map_final_aug @ self.image_encoder.proj

        image_features_list    = [image_features, image_features_aug]
        feature_map_list       = [feature_map_layer, feature_map_layer_aug]
        feature_map_final_list = [feature_map_final, feature_map_final_aug]

        # ── Group 2: frozen block + parallel cross-adapter (Appendix D) ──
        for i in range(self.num_layers - self.num_layers_final):
            cross_adapted = self.pseudo_forward(feature_map_list, i, is_last=False)
            block_idx = 11 - self.num_layers + i + 1
            for j in range(len(feature_map_list)):
                feature_map_list[j] = (
                    self.image_encoder.transformer.resblocks[block_idx](feature_map_list[j])
                    + cross_adapted[j]
                )

        # ── Group 3: fully fine-tuned blocks + Pseudo Forward (Eq. 10) ──
        (feat_ft_last_tmp,
         feature_ft_list_layer11_v,
         feature_ft_list_layer10,
         feature_ft_list_layer9,
         feature_ft_list_layer8) = self.pseudo_forward(feature_map_list, None, is_last=True)

        # Clone before in-place modification by the is_final projection below
        feature_ft_list_last    = [t.clone() for t in feat_ft_last_tmp]
        feature_ft_list_layer11 = [t.clone() for t in feat_ft_last_tmp]

        # ── Project all feature maps into the CLIP output embedding space ──
        feature_ft_list           = self.pseudo_forward(feat_ft_last_tmp,          None, is_final=True)
        feature_ft_list_layer8    = self.pseudo_forward(feature_ft_list_layer8,    None, is_final=True)
        feature_ft_list_layer9    = self.pseudo_forward(feature_ft_list_layer9,    None, is_final=True)
        feature_ft_list_layer10   = self.pseudo_forward(feature_ft_list_layer10,   None, is_final=True)
        feature_ft_list_layer11_v = self.pseudo_forward(feature_ft_list_layer11_v, None, is_final=True)

        # ── Normalize all features ──
        for i in range(len(feature_ft_list)):
            feature_ft_list[i]           = feature_ft_list[i]           / feature_ft_list[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer8[i]    = feature_ft_list_layer8[i]    / feature_ft_list_layer8[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer9[i]    = feature_ft_list_layer9[i]    / feature_ft_list_layer9[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer10[i]   = feature_ft_list_layer10[i]   / feature_ft_list_layer10[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer11[i]   = feature_ft_list_layer11[i]   / feature_ft_list_layer11[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer11_v[i] = feature_ft_list_layer11_v[i] / feature_ft_list_layer11_v[i].norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale         = self.logit_scale.exp()
        image_features_list = [image_features_list[i] / image_features_list[i].norm(dim=-1, keepdim=True)
                                for i in range(len(image_features_list))]
        text_features       = text_features / text_features.norm(dim=-1, keepdim=True)

        # Eq. (12): logits = α · logits_ft + (1 − α) · logits_zs
        logits_zs_list  = [logit_scale * image_features_list[i] @ text_features.t()
                           for i in range(len(image_features_list))]
        logits_ft_list  = [logit_scale * feature_ft_list[i][:, 0, :] @ text_features.t()
                           for i in range(len(feature_ft_list))]

        mean_logits_zs = sum(logits_zs_list) / len(logits_zs_list)
        mean_logits_ft = sum(logits_ft_list) / len(logits_ft_list)
        logits = self.alpha * mean_logits_ft + (1 - self.alpha) * mean_logits_zs

        # ──────────────────────────────────────────────────────────────────────
        # Manifold Alignment Regularization (MAR, Sec. 3.1)
        # ──────────────────────────────────────────────────────────────────────

        # Project fine-tuned token features for Gram matrix comparison
        for i in range(len(feature_ft_list_last)):
            feature_ft_list_last[i] = feature_ft_list_last[i].permute(1, 0, 2)
            feature_ft_list_last[i] = self.pseudo_forward.ln_post(feature_ft_list_last[i])
            feature_ft_list_last[i] = feature_ft_list_last[i] @ self.pseudo_forward.visual_proj
            feature_ft_list_last[i]   = feature_ft_list_last[i]   / feature_ft_list_last[i].norm(dim=-1, keepdim=True)
            feature_map_final_list[i] = feature_map_final_list[i] / feature_map_final_list[i].norm(dim=-1, keepdim=True)

        # L_MAR^local (Eq. 6): token-level (intra-sample) Gram matrix alignment
        gram_raw_v1 = torch.matmul(feature_map_final_list[0], feature_map_final_list[0].transpose(1, 2))
        gram_raw_v2 = torch.matmul(feature_map_final_list[1], feature_map_final_list[1].transpose(1, 2))
        gram_ft_v1  = torch.matmul(feature_ft_list_last[0],   feature_ft_list_last[0].transpose(1, 2))
        gram_ft_v2  = torch.matmul(feature_ft_list_last[1],   feature_ft_list_last[1].transpose(1, 2))
        mar_local_loss = (F.l1_loss(gram_raw_v1, gram_ft_v1, reduction='mean') +
                          F.l1_loss(gram_raw_v2, gram_ft_v2, reduction='mean'))

        # L_MAR^global (Eq. 5): batch-level [CLS] Gram matrix alignment
        gram_cls_raw_v1 = image_features_list[0] @ image_features_list[0].transpose(0, 1)
        gram_cls_raw_v2 = image_features_list[1] @ image_features_list[1].transpose(0, 1)
        gram_cls_ft_v1  = feature_ft_list[0][:, 0, :] @ feature_ft_list[0][:, 0, :].transpose(0, 1)
        gram_cls_ft_v2  = feature_ft_list[1][:, 0, :] @ feature_ft_list[1][:, 0, :].transpose(0, 1)
        mar_global_loss = (F.l1_loss(gram_cls_raw_v1, gram_cls_ft_v1, reduction='mean') +
                           F.l1_loss(gram_cls_raw_v2, gram_cls_ft_v2, reduction='mean'))

        return (logits, logits_zs_list, logits_ft_list, feature_ft_list,
                mar_local_loss, mar_global_loss,
                feature_ft_list_layer8, feature_ft_list_layer9,
                feature_ft_list_layer10, feature_ft_list_layer11,
                feature_ft_list_layer11_v, image_features_list)

    def forward_test(self, image):
        """Evaluation forward — single view, no Pseudo Forward needed."""
        text_features = self.text_features
        early_key = 'transformer.resblocks.{}'.format(11 - self.num_layers)

        with torch.no_grad():
            image_features, feat_dict = self.feat_extractor_test(image.type(self.dtype))
            feature_map = feat_dict[early_key]

        # Group 2: cross-adapter path
        for i in range(self.num_layers - self.num_layers_final):
            cross_adapted = self.pseudo_forward.forward_test(feature_map, i, is_last=False)
            block_idx = 11 - self.num_layers + i + 1
            feature_map = (
                self.image_encoder.transformer.resblocks[block_idx](feature_map)
                + cross_adapted
            )

        # Group 3: fully fine-tuned blocks
        feature_map = self.pseudo_forward.forward_test(feature_map, None, is_last=True)
        feature_ft  = self.pseudo_forward.forward_test(feature_map, None, is_final=True)

        # Normalize
        feature_ft     = feature_ft     / feature_ft.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_zs   = logit_scale * image_features @ text_features.t()
        logits_ft   = logit_scale * feature_ft      @ text_features.t()

        # Eq. (12): logits = α · logits_ft + (1 − α) · logits_zs
        return self.alpha * logits_ft + (1 - self.alpha) * logits_zs


# ──────────────────────────────────────────────────────────────────────────────
# MPS-Tuning Trainer
# ──────────────────────────────────────────────────────────────────────────────

@TRAINER_REGISTRY.register()
class MPSTuning_ClaudeRefined(TrainerX):
    """
    Trainer for Manifold-Preserving and Sculpting Tuning (MPS-Tuning).

    Overall training loss (Eq. 13):
        L = L_CE  +  λ1 · L_MAR  +  λ2 · L_HMS
    """

    def build_model(self):
        cfg        = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()
        self.scaler = GradScaler()

        print('Building MPS-Tuning model')
        self.model = MPSTuningCLIP(cfg, classnames, clip_model)

        print('Freezing image encoder and text encoder')
        for name, param in self.model.named_parameters():
            if 'CNN_Adapter' not in name and 'pseudo_forward' not in name:
                param.requires_grad_(False)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        params     = list(self.model.pseudo_forward.parameters())
        self.optim = build_optimizer(self.model.pseudo_forward, cfg.OPTIM, params)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('pseudo_forward', self.model.pseudo_forward, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

        # ── HMS loss functions — temperature τ′ is annealed during training (Appendix C) ──
        temp_output = 0.07
        self.hms_loss_fn = SupConLossWithText(
            temperature=temp_output,
            contrast_mode='all',
            base_temperature=temp_output,
        )
        temp_layer11 = 0.07
        self.hms_loss_fn_l11 = SupConLossWithText(
            temperature=temp_layer11,
            contrast_mode='all',
            base_temperature=temp_layer11,
        )
        temp_layer10 = 0.07
        self.hms_loss_fn_l10 = SupConLossWithText(
            temperature=temp_layer10,
            contrast_mode='all',
            base_temperature=temp_layer10,
        )

    def forward_backward(self, batch):
        image, image1, label = self.parse_batch_train(batch)
        optim  = self.optim
        scaler = self.scaler

        with torch.cuda.amp.autocast(enabled=True):
            (output, logits_zs_list, logits_ft_list, feature_ft_list,
             mar_local_loss, mar_global_loss,
             feature_ft_list_layer8, feature_ft_list_layer9,
             feature_ft_list_layer10, feature_ft_list_layer11,
             feature_ft_list_layer11_v,
             image_features_list) = self.model(image, image1)

            # L_CE — cross-entropy on combined logits (Eq. 13)
            ce_loss = F.cross_entropy(output, label)

            # ── Dynamic temperature annealing for HMS τ′ (Appendix C) ──
            if self.cfg.DATASET.NAME in [
                "OxfordPets", "Food101", "DescribableTextures", "EuroSAT", "UCF101"
            ]:
                cur_temp = get_annealed_temperature(
                    self.epoch, self.max_epoch,
                    initial_temp=0.5, final_temp=0.07, strategy='cosine',
                )
            else:
                cur_temp = get_annealed_temperature(
                    self.epoch, self.max_epoch,
                    initial_temp=0.1, final_temp=0.05, strategy='cosine',
                )
            cur_temp_layer10 = cur_temp * 2

            self.hms_loss_fn.temperature         = cur_temp
            self.hms_loss_fn.base_temperature     = cur_temp
            self.hms_loss_fn_l11.temperature      = cur_temp
            self.hms_loss_fn_l11.base_temperature = cur_temp
            self.hms_loss_fn_l10.temperature      = cur_temp_layer10
            self.hms_loss_fn_l10.base_temperature = cur_temp_layer10

            # ── L_HMS — Hierarchical Manifold Sculpting (Eq. 11) ──
            text_features = self.model.text_features
            text_labels   = torch.arange(len(text_features)).cuda()

            # Output-layer HMS (layer-wise decay weight = 1.0, Eq. 22)
            all_feats_output = torch.stack(feature_ft_list, dim=0)    # [V, B, S, D]
            all_feats_output = all_feats_output[:, :, 0, :]            # [V, B, D]  CLS token
            all_feats_output = all_feats_output.permute(1, 0, 2)       # [B, V, D]
            hms_loss_output  = self.hms_loss_fn(
                all_feats_output,
                labels=label,
                text_features=text_features,
                text_labels=text_labels,
            )

            # Layer-11 HMS (layer-wise decay weight = 1.0, Eq. 22)
            all_feats_l11 = torch.stack(feature_ft_list_layer11_v, dim=0)
            all_feats_l11 = all_feats_l11[:, :, 0, :]
            all_feats_l11 = all_feats_l11.permute(1, 0, 2)
            hms_loss_l11  = self.hms_loss_fn_l11(
                all_feats_l11,
                labels=label,
                text_features=text_features,
                text_labels=text_labels,
            )

            # Layer-10 HMS (layer-wise decay weight = 0.5, Eq. 22)
            all_feats_l10 = torch.stack(feature_ft_list_layer10, dim=0)
            all_feats_l10 = all_feats_l10[:, :, 0, :]
            all_feats_l10 = all_feats_l10.permute(1, 0, 2)
            hms_loss_l10  = self.hms_loss_fn_l10(
                all_feats_l10,
                labels=label,
                text_features=text_features,
                text_labels=text_labels,
            )

            # Eq. (11) + layer-wise decay weights from Eq. (22): w_L=1, w_l = 0.5·w_{l+1}
            hms_layer_weights = [1, 1, 0.5, 0.25, 0.125]
            hms_loss = (hms_loss_output * hms_layer_weights[0] +
                        hms_loss_l11    * hms_layer_weights[1] +
                        hms_loss_l10    * hms_layer_weights[2])

            # ── λ1: per-dataset MAR weight (Appendix C) ──
            if self.cfg.DATASET.NAME in [
                "Caltech101", "StanfordCars", "EuroSAT", "FGVCAircraft",
                "DescribableTextures", "OxfordFlowers", "UCF101"
            ]:
                lambda_mar = 0.5
            elif self.cfg.DATASET.NAME in ["Food101", "SUN397", "ImageNet", "OxfordPets"]:
                lambda_mar = 2
            else:
                raise ValueError(
                    f"Unknown dataset '{self.cfg.DATASET.NAME}' for lambda_mar (λ1)"
                )

            # L_MAR = L_MAR^local + L_MAR^global  (Eq. 7)
            mar_loss = mar_local_loss + mar_global_loss

            # Eq. (13): L = L_CE + λ1·L_MAR + λ2·L_HMS  (λ2 fixed at 0.1)
            loss = ce_loss + mar_loss * lambda_mar + hms_loss * 0.1

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_summary = {
            'loss':            loss.item(),
            'ce_loss':         ce_loss.item(),
            'hms_loss':        hms_loss.item(),
            'mar_local_loss':  mar_local_loss.item(),
            'mar_global_loss': mar_global_loss.item(),
            'acc':             compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def run_epoch(self):
        self.set_model_mode("train")
        losses     = MetricMeter()
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        self.num_batches = len(self.train_loader_x_2view)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x_2view):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq        = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain  = self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(seconds=int(batch_time.avg * nb_remain)))

                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def after_epoch(self):
        pass

    def parse_batch_train(self, batch):
        input  = batch['img'].to(self.device)
        input1 = batch['img1'].to(self.device)
        label  = batch['label'].to(self.device)
        return input, input1, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print('Note that load_model() is skipped as no pretrained model is given')
            return

        names      = self.get_model_names()
        model_file = 'model-best.pth.tar' if epoch is None else f'model.pth.tar-{epoch}'

        for name in names:
            model_path          = osp.join(directory, name, model_file)
            self.model_path_tmp = model_path

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch      = checkpoint['epoch']

            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """Standard evaluation pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split       = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model.forward_test(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)

        return list(results.values())[0]