import torch
import torch.nn as nn 

from .modeling import (IColoriTHintToken, IColoriTHintTokenLAB,
                          IColoriTHintTokenAB, IColoriTHintTokenLABTranshenc,
                          IColoriTHintTokenLABNOMASK, IColoriTHintTokenLABSmooth)
                            
from timm.models.registry import register_model
from functools import partial

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

# 09 12
@register_model
def icoloritv2lab_base_patch16_224_henc6_smooth(pretrained=False, **kwargs):
    model = IColoriTHintTokenLABSmooth(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        henc_depth=6,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

##### 폐기 
@register_model
def icoloritv2lab_base_patch16_224(pretrained=False, **kwargs):
    model = IColoriTHintTokenLAB(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        henc_depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2lab_base_patch16_224_henc6(pretrained=False, **kwargs):
    model = IColoriTHintTokenLAB(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        henc_depth=6,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2lab_base_patch16_224_henc6_nomask(pretrained=False, **kwargs):
    model = IColoriTHintTokenLABNOMASK(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        henc_depth=6,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2lab_base_patch16_224_henc6_transenc(pretrained=False, **kwargs):
    model = IColoriTHintTokenLABTranshenc(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        henc_depth=6,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icoloritv2lab_tiny_patch16_224(pretrained=False, **kwargs):
    model = IColoriTHintTokenLAB(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        depth=12,
        henc_depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2lab_tiny_patch16_224_henc6(pretrained=False, **kwargs):
    model = IColoriTHintTokenLAB(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        depth=12,
        henc_depth=6,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2_tiny_patch16_224(pretrained=False, **kwargs):
    model = IColoriTHintToken(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        depth=12,
        henc_depth=6,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2ab_base_patch16_224(pretrained=False, **kwargs):
    model = IColoriTHintTokenAB(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        henc_depth=6,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2ab_tiny_patch16_224(pretrained=False, **kwargs):
    model = IColoriTHintTokenAB(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        depth=12,
        henc_depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def icoloritv2tranhenc_tiny_patch16_224(pretrained=False, **kwargs):
    model = IColoriTHintTokenLABTranshenc(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        depth=12,
        henc_depth=6,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


