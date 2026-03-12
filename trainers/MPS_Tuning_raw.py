import os.path as osp
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

import time
import datetime
from collections import OrderedDict

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from typing import Dict, Iterable, Callable
import copy
import json
from tqdm import tqdm

from trainers.supcon_wtext import SupConLossWithText

_tokenizer = _Tokenizer()

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_model = clip_model.cuda()
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            # print("class_embeddings.shape: ", class_embeddings.shape)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=0).cuda()
    return clip_weights

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
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
    "Skin40": "a photo of a {}.",
    "galaxy": "a photo of a {}.",
    "IP102": "a photo of a {}.",
    "NWPU_RESISC45": "a centered satellite photo of {}.",
    "RFMiD": "a fundus image of {}.",
    "TCGA12": "a photo of a {}.",
    "NEU_CLS": "a photo of a hot-rolled steel plate with {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model



class feature_map_extractor(nn.Module):
    def __init__(self, model, layer_name):
        super(feature_map_extractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self._features = {layer: torch.empty(0) for layer in layer_name}

        for layer_id in layer_name:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn
    
    def forward(self, x):
        # _ = self.model(x)
        # return self._features
        image_features = self.model(x)
        _features = self._features
        self._features = {layer: torch.empty(0) for layer in self.layer_name}
        return image_features, _features


cafo_dict = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet",
    "ImageNetR": "imagenet",
    "ImageNetSketch": "imagenet",
    "ImageNetV2": "imagenet",
}

"""
Raw CLIP model: Transformer Block -> ... -> Transformer Block -> LayerNorm -> Projection
Adapted CLIP model: 
    Transformer Block -> A
    A + innerAdapter(A) -> B -> next Transformer Block -> C
    A + crossAdapter(A) -> D
    C = C + D -> LayerNorm -> Projection
"""

def custom_cross_entropy(logits1, logits2):
    """
    Calculate the cross entropy loss between two sets of logits.
    
    Args:
        logits1 (torch.Tensor): The original distribution logits.
        logits2 (torch.Tensor): The target distribution logits.
        
    Returns:
        torch.Tensor: The computed cross entropy loss.
    """
    # Convert logits2 to probability distribution using softmax
    target_distribution = F.softmax(logits2, dim=1)
    
    # Compute log probabilities for logits1
    log_probabilities = F.log_softmax(logits1, dim=1)
    
    # Calculate the cross entropy loss
    cross_entropy_loss = -torch.sum(target_distribution * log_probabilities, dim=1).mean()
    
    return cross_entropy_loss

class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        
        if "ImageNet" not in self.cfg.DATASET.NAME:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to('cuda')
            self.clip_model = self.clip_model.cuda()
            text_features = self.clip_model.encode_text(prompts)
        else:
            temp_list = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            text_features = []
            for temp in temp_list:
                prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts])
                prompts = prompts.to('cuda')
                self.clip_model = self.clip_model.cuda()
                text_features_temp = self.clip_model.encode_text(prompts)
                text_features.append(text_features_temp)
            text_features = torch.stack(text_features, dim=0).mean(dim=0)
        return text_features

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_weight(self, x: torch.Tensor):  # ADDED
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
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

    # only get x
    def forward_x(self, x: torch.Tensor):

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
    
    # only get v
    def forward_v(self, x: torch.Tensor):
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
    

import math
def get_annealed_temperature(current_epoch, total_epochs, initial_temp, final_temp, strategy='cosine'):
    if current_epoch >= total_epochs:
        return final_temp
    
    progress = current_epoch / total_epochs
    
    if strategy == 'linear':
        return initial_temp - progress * (initial_temp - final_temp)
    elif strategy == 'cosine':
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return final_temp + (initial_temp - final_temp) * cosine_decay
    else:
        raise ValueError("Unknown strategy")


class container(nn.Module):
    def __init__(self, ln_post, visual_proj, visual, num_layers, num_layers_final):
        super(container, self).__init__()

        self.num_layers = num_layers
        self.num_layers_final = num_layers_final
        
        self.ln_post = copy.deepcopy(ln_post)
        self.visual_proj = copy.deepcopy(visual_proj)

        # create innerAdapter, each adapter is a linear layer
        dtype = torch.float32
        self.crossAdapter = nn.ModuleList([nn.Linear(768, 768, bias=False, dtype=dtype) for _ in range(num_layers_final, num_layers)])

        self.clip_vit_layer8 = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer9 = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer10 = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer11 = ResidualAttentionBlock(768, 12)
        self.clip_vit_layer8.load_state_dict(visual.transformer.resblocks[8].state_dict())
        self.clip_vit_layer9.load_state_dict(visual.transformer.resblocks[9].state_dict())
        self.clip_vit_layer10.load_state_dict(visual.transformer.resblocks[10].state_dict())
        self.clip_vit_layer11.load_state_dict(visual.transformer.resblocks[11].state_dict())

        # 0-init
        for i in range(num_layers-num_layers_final):
            nn.init.zeros_(self.crossAdapter[i].weight)
        

    def forward(self, feature_map_list, layer_idx, is_last=False, is_final=False):

        if is_last:
            feature_map_list_layer8 = []
            feature_map_list_layer9 = []
            feature_map_list_layer10 = []
            feature_map_list_layer11 = []
            feature_map_list_layer11_output = []
            for i in range(len(feature_map_list)):
                layer8_o, _, _, layer8_v = self.clip_vit_layer8(feature_map_list[i])

                layer9_o, _, _, layer9_v = self.clip_vit_layer9(layer8_o)
                layer8_v = self.clip_vit_layer9.forward_v(layer8_v)

                layer10_o, _, _, layer10_v = self.clip_vit_layer10(layer9_o)
                layer9_v = self.clip_vit_layer10.forward_v(layer9_v)
                layer8_v = self.clip_vit_layer10.forward_v(layer8_v)

                layer11_o, _, _, layer11_v = self.clip_vit_layer11(layer10_o)
                layer10_v = self.clip_vit_layer11.forward_v(layer10_v)
                layer9_v = self.clip_vit_layer11.forward_v(layer9_v)
                layer8_v = self.clip_vit_layer11.forward_v(layer8_v)
                feature_map_list_layer8.append(layer8_v)
                feature_map_list_layer9.append(layer9_v)
                feature_map_list_layer10.append(layer10_v)
                feature_map_list_layer11.append(layer11_v)
                feature_map_list_layer11_output.append(layer11_o)

            return feature_map_list_layer11_output, feature_map_list_layer11, feature_map_list_layer10, feature_map_list_layer9, feature_map_list_layer8
        
        if is_final:
            for i in range(len(feature_map_list)):
                feature_map_list[i] = feature_map_list[i].permute(1, 0, 2)
                feature_map_list[i] = self.ln_post(feature_map_list[i])
                feature_map_list[i] = feature_map_list[i] @ self.visual_proj
                
            return feature_map_list

        else:
            crossadapted_feature_map_list = []
            for i in range(len(feature_map_list)):
                crossadapted_feature_map = self.crossAdapter[layer_idx](feature_map_list[i])
                crossadapted_feature_map_list.append(crossadapted_feature_map)

            return crossadapted_feature_map_list

    def forward_test(self, feature_map, layer_idx, is_last=False, is_final=False):
        if is_last:
            layer8_o = self.clip_vit_layer8.forward_x(feature_map)
            layer9_o = self.clip_vit_layer9.forward_x(layer8_o)
            layer10_o = self.clip_vit_layer10.forward_x(layer9_o)
            layer11_o = self.clip_vit_layer11.forward_x(layer10_o)
            return layer11_o
        
        if is_final:
            feature_map = feature_map.permute(1, 0, 2)
            feature_map = self.ln_post(feature_map[:, 0, :])
            feature_map = feature_map @ self.visual_proj
            return feature_map
            
        else:
            crossadapted_feature_map = self.crossAdapter[layer_idx](feature_map)
            return crossadapted_feature_map
        

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        with torch.no_grad():
            if "ImageNet" not in cfg.DATASET.NAME:
                temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
                prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts])
                prompts = prompts.to('cuda')
                clip_model = clip_model.cuda()
                text_features = clip_model.encode_text(prompts)
            else:
                temp_list = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
                text_features = []
                for temp in temp_list:
                    prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                    prompts = torch.cat([clip.tokenize(p) for p in prompts])
                    prompts = prompts.to('cuda')
                    clip_model = clip_model.cuda()
                    text_features_temp = clip_model.encode_text(prompts)
                    text_features.append(text_features_temp)
                text_features = torch.stack(text_features, dim=0).mean(dim=0)

            self.text_feature_test = text_features.cuda()
            self.text_feature = text_features.cuda()

            clip_model = clip_model.cpu()

        self.text_encoder = TextEncoder(cfg, classnames, clip_model)

        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.num_layers = 8
        self.num_layers_final = 4

        self.feature_extractor1 = feature_map_extractor(self.image_encoder, ['transformer.resblocks.{}'.format(11-self.num_layers), 'transformer.resblocks.11'])
        self.feature_extractor2 = feature_map_extractor(self.image_encoder, ['transformer.resblocks.{}'.format(11-self.num_layers), 'transformer.resblocks.11'])
        self.feature_extractor_test = feature_map_extractor(self.image_encoder, ['transformer.resblocks.{}'.format(11-self.num_layers)])
        self.finetune_container = container(clip_model.visual.ln_post, 
                                            clip_model.visual.proj,
                                            clip_model.visual,
                                            num_layers=self.num_layers,
                                            num_layers_final=self.num_layers_final)

        self.alpha = 0.3

    def forward(self, image, image1):
        text_features = self.text_feature

        with torch.no_grad():
            image_features, feature_map_layer_dict = self.feature_extractor1(image.type(self.dtype))
            image_features_aug, feature_map_layer_aug_dict = self.feature_extractor2(image1.type(self.dtype))
            feature_map_layer = feature_map_layer_dict['transformer.resblocks.{}'.format(11-self.num_layers)]
            feature_map_layer_aug = feature_map_layer_aug_dict['transformer.resblocks.{}'.format(11-self.num_layers)]
            
            feature_map_final = feature_map_layer_dict['transformer.resblocks.11']
            feature_map_final_aug = feature_map_layer_aug_dict['transformer.resblocks.11']
            feature_map_final = feature_map_final.permute(1, 0, 2)
            feature_map_final_aug = feature_map_final_aug.permute(1, 0, 2)
            feature_map_final = self.image_encoder.ln_post(feature_map_final)
            feature_map_final_aug = self.image_encoder.ln_post(feature_map_final_aug)
            feature_map_final = feature_map_final @ self.image_encoder.proj
            feature_map_final_aug = feature_map_final_aug @ self.image_encoder.proj

        image_features_list = [image_features, image_features_aug]
        feature_map_list = [feature_map_layer, feature_map_layer_aug]
        feature_map_final_list = [feature_map_final, feature_map_final_aug]

        for i in range(self.num_layers-self.num_layers_final):
            crossadapted_feature_map = self.finetune_container(feature_map_list, i, is_last=False)
            for j in range(len(feature_map_list)):
                feature_map_list[j] = self.image_encoder.transformer.resblocks[11-self.num_layers+i+1](feature_map_list[j])
                feature_map_list[j] = feature_map_list[j] + crossadapted_feature_map[j]

        feature_ft_list_last_tmp, feature_ft_list_layer11_v, feature_ft_list_layer10, feature_ft_list_layer9, feature_ft_list_layer8 = self.finetune_container(feature_map_list, None, is_last=True)
        feature_ft_list_last = []
        feature_ft_list_layer11 = []
        for i in range(len(feature_ft_list_last_tmp)):
            feature_ft_list_last.append(feature_ft_list_last_tmp[i].clone())
            feature_ft_list_layer11.append(feature_ft_list_last_tmp[i].clone())
        feature_ft_list = self.finetune_container(feature_ft_list_last_tmp, None, is_last=False, is_final=True)
        
        feature_ft_list_layer8 = self.finetune_container(feature_ft_list_layer8, None, is_last=False, is_final=True)
        feature_ft_list_layer9 = self.finetune_container(feature_ft_list_layer9, None, is_last=False, is_final=True)
        feature_ft_list_layer10 = self.finetune_container(feature_ft_list_layer10, None, is_last=False, is_final=True)
        feature_ft_list_layer11_v = self.finetune_container(feature_ft_list_layer11_v, None, is_last=False, is_final=True)
        
        for i in range(len(feature_ft_list)):
            feature_ft_list[i] = feature_ft_list[i] / feature_ft_list[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer8[i] = feature_ft_list_layer8[i] / feature_ft_list_layer8[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer9[i] = feature_ft_list_layer9[i] / feature_ft_list_layer9[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer10[i] = feature_ft_list_layer10[i] / feature_ft_list_layer10[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer11[i] = feature_ft_list_layer11[i] / feature_ft_list_layer11[i].norm(dim=-1, keepdim=True)
            feature_ft_list_layer11_v[i] = feature_ft_list_layer11_v[i] / feature_ft_list_layer11_v[i].norm(dim=-1, keepdim=True)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        image_features_list = [image_features_list[i] / image_features_list[i].norm(dim=-1, keepdim=True) for i in range(len(image_features_list))]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        raw_logits_list = [logit_scale * image_features_list[i] @ text_features.t() for i in range(len(image_features_list))]
        logits_ft_list = [logit_scale * feature_ft_list[i][:, 0, :] @ text_features.t() for i in range(len(feature_ft_list))]

        alpha = self.alpha
        mean_raw_logits = sum(raw_logits_list) / len(raw_logits_list)
        mean_logits_ft = sum(logits_ft_list) / len(logits_ft_list)
        logits = alpha * mean_logits_ft + (1 - alpha) * mean_raw_logits
        
        # optimize token similarity paradigm between feature_ft_list_last and feature_map_final_list
        # get similarity matrix
        for i in range(len(feature_ft_list_last)):
            feature_ft_list_last[i] = feature_ft_list_last[i].permute(1, 0, 2)
            feature_ft_list_last[i] = self.finetune_container.ln_post(feature_ft_list_last[i])
            feature_ft_list_last[i] = feature_ft_list_last[i] @ self.finetune_container.visual_proj
            feature_ft_list_last[i] = feature_ft_list_last[i] / feature_ft_list_last[i].norm(dim=-1, keepdim=True)
            feature_map_final_list[i] = feature_map_final_list[i] / feature_map_final_list[i].norm(dim=-1, keepdim=True)
        similarity_matrix_raw_1 = torch.matmul(feature_map_final_list[0], feature_map_final_list[0].transpose(1, 2))
        similarity_matrix_raw_2 = torch.matmul(feature_map_final_list[1], feature_map_final_list[1].transpose(1, 2))
        similarity_matrix_ft_1 = torch.matmul(feature_ft_list_last[0], feature_ft_list_last[0].transpose(1, 2))
        similarity_matrix_ft_2 = torch.matmul(feature_ft_list_last[1], feature_ft_list_last[1].transpose(1, 2))
        # get similarity loss
        similarity_loss_1 = F.l1_loss(similarity_matrix_raw_1, similarity_matrix_ft_1, reduction='mean')
        similarity_loss_2 = F.l1_loss(similarity_matrix_raw_2, similarity_matrix_ft_2, reduction='mean')
        similarity_loss = similarity_loss_1 + similarity_loss_2
        
        # calculate similarity in final feature in batch-level
        feat_sim_matrix_raw_1 = image_features_list[0] @ image_features_list[0].transpose(0, 1)
        feat_sim_matrix_raw_2 = image_features_list[1] @ image_features_list[1].transpose(0, 1)
        feat_sim_matrix_ft_1 = feature_ft_list[0][:, 0, :] @ feature_ft_list[0][:, 0, :].transpose(0, 1)
        feat_sim_matrix_ft_2 = feature_ft_list[1][:, 0, :] @ feature_ft_list[1][:, 0, :].transpose(0, 1)
        feat_sim_loss_1 = F.l1_loss(feat_sim_matrix_raw_1, feat_sim_matrix_ft_1, reduction='mean')
        feat_sim_loss_2 = F.l1_loss(feat_sim_matrix_raw_2, feat_sim_matrix_ft_2, reduction='mean')
        feat_sim_loss = feat_sim_loss_1 + feat_sim_loss_2
        

        return logits, raw_logits_list, logits_ft_list, feature_ft_list, similarity_loss, feat_sim_loss, feature_ft_list_layer8, feature_ft_list_layer9, feature_ft_list_layer10, feature_ft_list_layer11, feature_ft_list_layer11_v, \
            image_features_list


    def forward_test(self, image):
        text_features = self.text_feature_test

        with torch.no_grad():
            image_features, feature_map_layer_dict = self.feature_extractor_test(image.type(self.dtype))
            feature_map = feature_map_layer_dict['transformer.resblocks.{}'.format(11-self.num_layers)]
            
        # Sequentially apply adapter and transformer blocks
        for i in range(self.num_layers - self.num_layers_final):
            crossadapted_feature_map = self.finetune_container.forward_test(feature_map, i, is_last=False)
            feature_map = self.image_encoder.transformer.resblocks[11 - self.num_layers + i + 1](feature_map)
            feature_map = feature_map + crossadapted_feature_map

        # Pass through final layers and projection
        feature_map_last = self.finetune_container.forward_test(feature_map, None, is_last=True)
        feature_ft = self.finetune_container.forward_test(feature_map_last, None, is_last=False, is_final=True)
        
        # Normalize features
        feature_ft = feature_ft / feature_ft.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate logits
        logit_scale = self.logit_scale.exp()
        raw_logits = logit_scale * image_features @ text_features.t()
        logits_ft = logit_scale * feature_ft @ text_features.t()

        # Combine logits
        alpha = self.alpha
        logits = alpha * logits_ft + (1 - alpha) * raw_logits
        
        return logits
        

@TRAINER_REGISTRY.register()
class MPSTuning(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()
        self.scaler = GradScaler()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'CNN_Adapter' not in name and 'finetune' not in name:
                param.requires_grad_(False)
        
        # print the number of trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        params = list(self.model.finetune_container.parameters())
        self.optim = build_optimizer(self.model.finetune_container, cfg.OPTIM, params)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('finetune_container', self.model.finetune_container, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)
            
        temp_all = 0.07
        self.supcon = SupConLossWithText(
            temperature=temp_all,
            contrast_mode='all',
            base_temperature=temp_all,
        )
        temp_layer11 = 0.07
        self.supcon_layer11 = SupConLossWithText(
            temperature=temp_layer11,
            contrast_mode='all',
            base_temperature=temp_layer11,
        )
        temp_layer10 = 0.07
        self.supcon_layer10 = SupConLossWithText(
            temperature=temp_layer10,
            contrast_mode='all',
            base_temperature=temp_layer10,
        )

    def forward_backward(self, batch):
        image, image1, label = self.parse_batch_train(batch)
        optim = self.optim
        scaler = self.scaler
        with torch.cuda.amp.autocast(enabled=True):
            output, raw_output_list, ft_output_list, feature_ft_list, similarity_loss, feat_sim_loss, \
                feature_ft_list_layer8, feature_ft_list_layer9, feature_ft_list_layer10, feature_ft_list_layer11, feature_ft_list_layer11_v, \
                image_features_list = self.model(image, image1)
            
            ce_loss = F.cross_entropy(output, label)
            kl_loss = 0
            for output_idx in range(len(raw_output_list)):
                kl_loss += F.l1_loss(raw_output_list[output_idx], ft_output_list[output_idx])/len(raw_output_list)

                
            # # calculate the supervised contrastive loss
            text_features = self.model.text_feature
            text_labels = torch.arange(len(text_features)).cuda()

            # if dataset name is pets, food, dtd, eurosat, ucf, use dynamic temperature
            if self.cfg.DATASET.NAME in ["OxfordPets", "Food101", "DescribableTextures", "EuroSAT", "UCF101"]:
                cur_temp = get_annealed_temperature(
                    self.epoch, self.max_epoch, 
                    initial_temp=0.5, final_temp=0.07, strategy='cosine'
                )
                cur_temp_layer10 = cur_temp*2
            else:
                cur_temp = get_annealed_temperature(
                    self.epoch, self.max_epoch, 
                    initial_temp=0.1, final_temp=0.05, strategy='cosine'
                )
                cur_temp_layer10 = cur_temp*2
                
            # set temperature for supcon
            self.supcon.temperature = cur_temp
            self.supcon.base_temperature = cur_temp
            self.supcon_layer11.temperature = cur_temp
            self.supcon_layer11.base_temperature = cur_temp
            self.supcon_layer10.temperature = cur_temp_layer10
            self.supcon_layer10.base_temperature = cur_temp_layer10
            
            all_features = torch.stack(feature_ft_list, dim=0) # [view_num, batch_size, cls_num]
            all_features = all_features[:, :, 0, :] # [view_num, batch_size, cls_num]
            all_features = all_features.permute(1, 0, 2) # [batch_size, view_num, cls_num]
            supcon_loss_final = self.supcon(
                all_features,
                labels=label,
                text_features=text_features,
                text_labels=text_labels,
            )
            all_features_layer11 = torch.stack(feature_ft_list_layer11_v, dim=0) # [view_num, batch_size, cls_num]
            all_features_layer11 = all_features_layer11[:, :, 0, :] # [view_num, batch_size, cls_num]
            all_features_layer11 = all_features_layer11.permute(1, 0, 2) # [batch_size, view_num, cls_num]
            supcon_loss_layer11 = self.supcon_layer11(
                all_features_layer11,
                labels=label,
                text_features=text_features,
                text_labels=text_labels,
            )
            all_features_layer10 = torch.stack(feature_ft_list_layer10, dim=0) # [view_num, batch_size, cls_num]
            all_features_layer10 = all_features_layer10[:, :, 0, :] # [view_num, batch_size, cls_num]
            all_features_layer10 = all_features_layer10.permute(1, 0, 2) # [batch_size, view_num, cls_num]
            supcon_loss_layer10 = self.supcon_layer10(
                all_features_layer10,
                labels=label,
                text_features=text_features,
                text_labels=text_labels,
            )
            
            cupcon_alpha_list = [1, 1, 0.5, 0.25, 0.125]
            supcon_loss = supcon_loss_final*cupcon_alpha_list[0] + \
                supcon_loss_layer11*cupcon_alpha_list[1] + \
                supcon_loss_layer10*cupcon_alpha_list[2]
            
            
            if self.cfg.DATASET.NAME in ["Caltech101", "StanfordCars", "EuroSAT", "FGVCAircraft", "DescribableTextures", "OxfordFlowers", "UCF101"]:
                con_alpha = 0.5
            elif self.cfg.DATASET.NAME in ["Food101", "SUN397", "ImageNet", "OxfordPets"]:
                con_alpha = 2
            else:
                raise ValueError(
                    f"Unknown dataset {self.cfg.DATASET.NAME} for con_alpha"
                )
            loss = ce_loss + (similarity_loss + feat_sim_loss) * con_alpha + \
                supcon_loss*0.1

        # self.model_backward_and_update(loss)
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_summary = {
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'supcon_loss': supcon_loss.item(),
            'kl_loss': kl_loss.item(),
            'similarity_loss': similarity_loss.item(),
            'feat_sim_loss': feat_sim_loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x_2view)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x_2view):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def after_epoch(self):
        pass

    def parse_batch_train(self, batch):
        input = batch['img']
        input1 = batch['img1']
        label = batch['label']
        input = input.to(self.device)
        input1 = input1.to(self.device)
        label = label.to(self.device)
        return input, input1, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            self.model_path_tmp = model_path

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict, strict=False)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # use amp
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model.forward_test(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]