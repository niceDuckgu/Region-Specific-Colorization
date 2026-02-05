import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .common import Mlp, Attention, DropPath

### TODO HINT 

class MaskedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        # mask is not needed !!!!
        # if mask is not None:
        #     # Memory issue Repeat vs permute 
        #     # x = (x.permute(2,0,1)*mask).permute(1,2,0)  # not necessery from debbuger
        #     # x = x * mask.repeat(1,1,x.shape[-1])
        #     x = self.fc1(x)
        #     x = (x.permute(2,0,1)*mask).permute(1,2,0)
        #     # x = x * mask.repeat(1,1,x.shape[-1])
        #     x = self.act(x)
        #     # x = self.drop(x)
        #     # commit this for the orignal BERT implement
        #     x = self.fc2(x)
        #     # x = x * mask.repeat(1,1,x.shape[-1])
        #     x = self.drop(x)
        # else:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PadHintEmhed(nn.Module): 
    """
    Input: AB color, resized patches with size PXP, hint ratio list
    output: Embeeded value 
    """
    def __init__(self, max_hint=150, window_size=16, embed_dim=768, train=True):
        super().__init__()
        self.max_hint = max_hint
        # self.mask_mode = mask_mode
        self.p = 16
        self._train = train
        self.proj = nn.Linear(self.p**2+2, embed_dim)
        # self.pad_pos_emb = nn.Embedding(num_embeddings=1, embedding_dim=embed_dim).weight 
        # input shape: B*h*P*P 
        #  torch.Size([256, 196, 192]) # B* token length* emb dim 

    def forward(self, x, ab, **kwargs):
        # x: b*max_hint*1*P*P ab: b*max_hint*ab num_hint: b*max_hint_len

        assert x.dim() ==4 , \
            f"Input patch dimension ({x.dim()}) doesn't match model input (3 or 4) "

        B, C, H, W = x.shape
        x = rearrange(x, 'b h p1 p2 -> b h (p1 p2)', p1=self.p, p2=self.p)
        x = torch.cat([x,ab], dim=-1)
        x = self.proj(x)#.flatten(2).transpose(1, 2)
        # B,H, emb = x.shape
        return x

class LABPadHintEmhed(nn.Module): 
    def __init__(self, max_hint=150, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.max_hint = max_hint
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class LABMLPHintEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=6,
                 act_layer=nn.GELU, max_hint_len=150,  patch_size=16):
        super().__init__()
        
        self.padded_hint_embed = LABPadHintEmhed( 
                                    max_hint=max_hint_len, 
                                    patch_size=patch_size,
                                    embed_dim=embed_dim)
        # self.pad_embed = self.padded_hint_embed.pad_pos_emb

        self.depth = depth
        self.max_hint_len = max_hint_len

        self.blocks = nn.ModuleList([MaskedMlp(
            in_features=embed_dim, hidden_features=embed_dim, act_layer=act_layer)
            for i in range(depth)])
      
    def hint2mask(self, mask, max_hint_len, device='cuda'):
        # The input of the hint encoder is (HHHHPPPPP) s.t H: hint, P: padding
        # Therefore, for the M.T.E, we need to make the mask with size of B*(oL+mhL)
        num_hint = torch.sum(mask, dim=-1)
        mask = [0]*mask.shape[0]
        for idx, m in enumerate(num_hint):
            assert m.item()<=max_hint_len
            mask[idx] = torch.tensor([1]*(m.item()) + [0]*(max_hint_len-m.item()))
        return torch.stack(mask, dim=0).to(device)

    def forward(self,h):
        # Padded images, 
        B, T, C, H, W = h.shape
        h = rearrange(h, 'b h c p1 p2 -> (b h) c p1 p2', p1=H, p2=W)
        h = self.padded_hint_embed(h) # zero intput==> embedded (1* 768 vec)
        h = rearrange(h, '(b h) c emb -> b (h c) emb', b=B, h= T)
        # TODO REMOVE 08.26
        for blk in self.blocks:
            h = blk(h, None)

        B,N,C = h.shape

        return h

### ablation study (only ab color into hint enc)
#TODO 

class ABPadHintEmhed(nn.Module): 
    def __init__(self, max_hint=150, in_chans=2, embed_dim=768):
        super().__init__()
        self.max_hint = max_hint
        self.proj = nn.Linear(in_chans, embed_dim)

    def forward(self, x, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ABMLPHintEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=6,
                 act_layer=nn.GELU, max_hint_len=150,  patch_size=16):
        super().__init__()
        
        self.padded_hint_embed = ABPadHintEmhed(max_hint=max_hint_len, 
                                    in_chans=2,
                                    embed_dim=embed_dim)
        # self.pad_embed = self.padded_hint_embed.pad_pos_emb

        self.depth = depth
        self.max_hint_len = max_hint_len
        # K V 
      
      
        self.blocks = nn.ModuleList([MaskedMlp(
            in_features=embed_dim, hidden_features=embed_dim, act_layer=act_layer)
            for i in range(depth)])
      
    def hint2mask(self, mask, max_hint_len, device='cuda'):
        # The input of the hint encoder is (HHHHPPPPP) s.t H: hint, P: padding
        # Therefore, for the M.T.E, we need to make the mask with size of B*(oL+mhL)
        num_hint = torch.sum(mask, dim=-1)
        mask = [0]*mask.shape[0]
        for idx, m in enumerate(num_hint):
            assert m.item()<=max_hint_len
            mask[idx] = torch.tensor([1]*(m.item()) + [0]*(max_hint_len-m.item()))
        return torch.stack(mask, dim=0).to(device)

    def forward(self, hint_ab, mask):
        # Padded images, 
        _device = 'cuda' if hint_ab.device.type == 'cuda' else 'cpu'
        
        h = self.padded_hint_embed(hint_ab).permute(0,2,1) # B * T * EMB 
        #TODO REMOVE
        # if self.max_hint_len is not None:
        #     # TODO remove 
        #     padtoken_mask = self.hint2mask(mask, self.max_hint_len, _device).bool()
        #     for blk in self.blocks:
        #         h = blk(h, padtoken_mask)
        #     #TODO remove masking ==> null            
        #     h = (h.permute(2,0,1)*padtoken_mask).permute(1,2,0)
        # else:
        #     for blk in self.blocks:
        #         h = blk(h, None)
        for blk in self.blocks:
            h = blk(h, None)

        B,N,C = h.shape

        return h



class MLPHintEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=6,
                 act_layer=nn.GELU, max_hint_len=150,  patch_size=16):
        super().__init__()
        
        self.padded_hint_embed = PadHintEmhed(max_hint=max_hint_len, 
                                    window_size=patch_size,
                                    embed_dim=embed_dim)
        # self.pad_embed = self.padded_hint_embed.pad_pos_emb

        self.depth = depth
        self.max_hint_len = max_hint_len
        # K V 
      
      
        self.blocks = nn.ModuleList([MaskedMlp(
            in_features=embed_dim, hidden_features=embed_dim, act_layer=act_layer)
            for i in range(depth)])
      
    def hint2mask(self, mask, max_hint_len, device='cuda'):
        # The input of the hint encoder is (HHHHPPPPP) s.t H: hint, P: padding
        # Therefore, for the M.T.E, we need to make the mask with size of B*(oL+mhL)
        num_hint = torch.sum(mask, dim=-1)
        mask = [0]*mask.shape[0]
        for idx, m in enumerate(num_hint):
            assert m.item()<=max_hint_len
            mask[idx] = torch.tensor([1]*(m.item()) + [0]*(max_hint_len-m.item()))
        return torch.stack(mask, dim=0).to(device)

    def forward(self,hint_img, hint_ab, mask):
        # Padded images, 
        _device = 'cuda' if hint_img.device.type == 'cuda' else 'cpu'
        
        h = self.padded_hint_embed(hint_img, hint_ab)
        #TODO REMOVE
        # if self.max_hint_len is not None:
        #     # TODO remove
        #     padtoken_mask = self.hint2mask(mask, self.max_hint_len, _device).bool()
        #     for blk in self.blocks:
        #         h = blk(h, padtoken_mask)
        #     # TODO remove
        #     h = (h.permute(2,0,1)*padtoken_mask).permute(1,2,0)
        # else:
        #     for blk in self.blocks:
        #         h = blk(h, None)
        for blk in self.blocks:
            h = blk(h, None)

        B,N,C = h.shape

        return h

# 08 12

class HintAttenBlock(nn.Module): 
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn1 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        else:
            self.gamma_1, self.gamma_2 =  None, None

    def forward(self, x, mask = None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn1(self.norm1(x), mask))
            # x = (x.permute(2,0,1)*mask).permute(1,2,0)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            # x = (x.permute(2,0,1)*mask).permute(1,2,0)
#
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn1(self.norm1(x), mask))
            # x = (x.permute(2,0,1)*mask).permute(1,2,0)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            # x = (x.permute(2,0,1)*mask).permute(1,2,0)

        return x


class LABTransformerHintEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, patch_size=16, window_size=14, max_hint_len=150):
        super().__init__()
        
        self.padded_hint_embed = LABPadHintEmhed( 
                                    max_hint=max_hint_len, 
                                    patch_size=patch_size,
                                    embed_dim=dim)
        # self.pad_embed = self.padded_hint_embed.pad_pos_emb

        self.depth = depth
        self.max_hint_len = max_hint_len

        self.blocks = nn.ModuleList([HintAttenBlock(
            dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=0., attn_drop=0., drop_path=drop_path, norm_layer=norm_layer,
            init_values=init_values, use_rpb=use_rpb, window_size=window_size) for i in range(depth)])

    def hint2mask(self, mask, max_hint_len, device='cuda'):
        # The input of the hint encoder is (HHHHPPPPP) s.t H: hint, P: padding
        # Therefore, for the M.T.E, we need to make the mask with size of B*(oL+mhL)
        num_hint = torch.sum(mask, dim=-1)
        mask = [0]*mask.shape[0]
        for idx, m in enumerate(num_hint):
            assert m.item()<=max_hint_len
            mask[idx] = torch.tensor([1]*(m.item()) + [0]*(max_hint_len-m.item()))
        return torch.stack(mask, dim=0).to(device)

    def forward(self,h, mask):
        # Padded images, 
        _device = 'cuda' if h.device.type == 'cuda' else 'cpu'
        B, T, C, H, W = h.shape
        h = rearrange(h, 'b h c p1 p2 -> (b h) c p1 p2', p1=H, p2=W)
        h = self.padded_hint_embed(h)
        h = rearrange(h, '(b h) c emb -> b (h c) emb', b=B, h= T)

        # TODO REMOVE         
        # if self.max_hint_len is not None:
        #     #TODO remove
        #     padtoken_mask = self.hint2mask(mask, self.max_hint_len, _device).bool()

        #     for blk in self.blocks:
        #         h = blk(h, padtoken_mask)
        #     #TODO revmoe
        #     h = (h.permute(2,0,1)*padtoken_mask).permute(1,2,0)

        # else:
        #     for blk in self.blocks:
        #         h = blk(h, None)
        for blk in self.blocks:
            h = blk(h, None)

        B,N,C = h.shape

        return h
