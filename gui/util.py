
import torch
from PIL import Image
import numpy as np
import cv2

from torchvision.ops import masks_to_boxes
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize, Resize, Pad, InterpolationMode

def rollout(attentions, discard_ratio, head_fusion, token_idx):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, token_idx, :]
    mask = result[0, :, token_idx]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def show_mask_on_image(img, mask):
    # mask = (np.clip(mask, 0, mask.mean())) / (np.clip(mask, 0, mask.mean())).max()
    mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9, patch_size=16):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.patch_size = patch_size
        self.attention_layer_name = attention_layer_name
        self.attentions = []
        self.prev_token = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def control_attention(self, module, input, output):
        self.attentions.append(output.cpu())
        for idx in self.prev_token:
            output[:,:, idx,idx] = 1e10 
            output[:,:,idx,:] = output[:,:,idx,:].softmax(dim=-1)
        print(f'token num {self.prev_token}', module)
        return output

    def __call__(self, img_lab, bool_masked_pos, gt, mask_coords):
        hook_lists = []
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                layer_num = name.split('.')[1]
                # if int(layer_num) > 5:
                #     module.register_forward_hook(self.get_attention)
                hook_lists.append(module.register_forward_hook(self.get_attention))

        self.attentions = []
        with torch.no_grad():
            outputs = self.model(img_lab, bool_masked_pos)
        hook_lists = [i.remove() for i in hook_lists]
        B, C, H, W = img_lab.shape
        token_coords = mask_coords // self.patch_size
        token_coords = [token_coords[0]] # first tokoen 
        img_rgb = gt[0].permute(1, 2, 0)
        np_img = np.array(img_rgb.cpu())[:, :, ::-1]
        rollout_masks = []
        for token_coord in token_coords:
            token_idx = (token_coord[0] * (H / self.patch_size) + token_coord[1]).long()
            print(f'token idx {token_idx}')
            rollout_mask = rollout(self.attentions, self.discard_ratio, self.head_fusion, token_idx)
            rollout_mask = show_mask_on_image(np_img, rollout_mask)
            rollout_masks.append(rollout_mask)

        return rollout_masks, outputs
    def get_adj_token(self, token_dix, H, W):
        #  idx = (H/ self.patch_size).long()
         return None

    def control_mask(self, img_lab, bool_masked_pos, gt, mask_coords):
        B, C, H, W = img_lab.shape
        self.attentions = []

        token_coords = mask_coords // self.patch_size

        img_rgb = gt[0].permute(1, 2, 0)
        np_img = np.array(img_rgb.cpu())[:, :, ::-1]
        rollout_masks = []
        token_coord = token_coords[-1]
        token_idx = (token_coord[0] * (H / self.patch_size) + token_coord[1]).long()
        self.prev_token.append(token_idx.item())
        self.prev_token = list(set(self.prev_token))
        self.adjtoken_idx = self.get_adj_token(token_idx, H, W)
        bolck_names = [ # 'blocks.0.attn.attn_drop',
                        # 'blocks.1.attn.attn_drop',
                        # 'blocks.2.attn.attn_drop',
                        # 'blocks.3.attn.attn_drop',
                        # 'blocks.4.attn.attn_drop',
                        # 'blocks.5.attn.attn_drop',
                        'blocks.6.attn.attn_drop',
                        'blocks.7.attn.attn_drop',
                        'blocks.8.attn.attn_drop',
                        'blocks.9.attn.attn_drop',
                        'blocks.10.attn.attn_drop',
                        'blocks.11.attn.attn_drop']
        hook_lists = []
        for name, module in self.model.named_modules():
            # if self.attention_layer_name in name:
            if name in bolck_names:
                layer_num = name.split('.')[1]
                hook_lists.append(module.register_forward_hook(self.control_attention))
        with torch.no_grad():
            outputs = self.model(img_lab, bool_masked_pos)

        hook_lists = [i.remove()for i in hook_lists] 
        del(hook_lists)
        rollout_mask = rollout(self.attentions, self.discard_ratio, self.head_fusion, token_idx)
        rollout_mask = show_mask_on_image(np_img, rollout_mask)
        rollout_masks.append(rollout_mask)
        return rollout_masks, outputs



def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2xyz')
    # embed()
    return out


def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)

    # if(torch.sum(torch.isnan(rgb))>0):
    # print('xyz2rgb')
    # embed()
    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    # if(torch.sum(torch.isnan(out))>0):
    # print('xyz2lab')
    # embed()

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc

    # if(torch.sum(torch.isnan(out))>0):
    # print('lab2xyz')
    # embed()

    return out

def rgb2lab(rgb, l_cent=50, l_norm=100, ab_norm=110):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - l_cent) / l_norm
    ab_rs = lab[:, 1:, :, :] / ab_norm
    out = torch.cat((l_rs, ab_rs), dim=1)
    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2lab')
    # embed()
    return out


def lab2rgb(lab_rs, l_cent=50, l_norm=100, ab_norm=110):
    l = lab_rs[:, [0], :, :] * l_norm + l_cent
    ab = lab_rs[:, 1:, :, :] * ab_norm
    lab = torch.cat((l, ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
    # print('lab2rgb')
    # embed()
    return out

import os
import cv2
import math

import torch
import torch.nn.functional as F

import numpy as np 


def pad_and_crop_image(hint_img, x, y, pad_size=8):
    """
    Pads the image if bounding box coordinates are out of bounds and then crops it.
    The image is expected to be a PyTorch tensor.

    Parameters:
    - hint_img: The image tensor of shape [C, H, W].
    - x, y: Center coordinates for the cropping area.

    Returns:
    - Cropped image tensor according to the padded and corrected bounding box.
    """
    C, H, W = hint_img.shape

    # Compute bounding box coordinates
    xmin = x - pad_size
    xmax = x + pad_size
    ymin = y - pad_size
    ymax = y + pad_size

    hint_img = F.pad(hint_img, (pad_size, pad_size, pad_size, pad_size), 'reflect')

    # Update bounding box coordinates after padding
    xmin += pad_size
    xmax += pad_size
    ymin += pad_size
    ymax += pad_size

    # Crop the image
    cropped_img = hint_img[:, ymin:ymax, xmin:xmax]

    return cropped_img

def get_hint_loc(hint, hint_size=2, input_size=224):
    # hint shape L: H*W//hint_size**2 : hint loc
    coor = np.where(hint==0)
    x_coor = coor%np.array(input_size//hint_size) 
    y_coor = coor//np.array(input_size//hint_size)
    return [[x*hint_size,y*hint_size] for x, y in zip(x_coor[0],y_coor[0])]

def nulltoken_masking(mask):
    # index of null token=-1 ==> h h h, ..., m, m , ... , null toke 
    _mask = torch.sum(mask, dim=-1)==0
    _mask_ = mask.clone()
    _mask_= _mask_[_mask]
    _mask_[:,-1] = 1
    mask[_mask] = _mask_
    return mask    


def interactive_sampler(img_l, hintloc ,a, b, P=16,
                        input_size=224, hint_size=2 , center=False, fixmask=False,segmap=None, crop_size=8):
    """
    img_l: normlized l channe img
    hintloc => user pos (x,y)
    lc, rc : left top, right bottom [(xmin, ymin)], [(xmax, ymax)]
    a, b : color 
    P: patch size 
    """
    hint_gen = FixedCenterCroppedImageGeneratorFromLABHint(P=P, input_size=input_size, hint_size=hint_size , center=center, fixmask=fixmask, crop_size=crop_size)
    cimg, hint_mask = hint_gen.crop_img(img_l, hintloc, a, b)
    return cimg, hint_mask 

class FixedCenterCroppedImageGeneratorFromLABHint:
    # iColoriT manner hint encoder 
    def __init__(self, P, input_size, hint_size=2 , center=True, fixmask=False, crop_size=8):
        # center: if True: click points are center points of the rx, ry 
        #         else: click points and rx, ry are i.d
        self.P = P
        self.input_size = input_size
        self.token_num = (input_size//P)**2
        self.hint_size = hint_size
        self.center = center
        self.default_hint_len = 1
        self.fixmask = fixmask
        self.crop_size = crop_size
    def __call__(self, img_l, hint_loc, a, b):
        return self.crop_img(img_l, hint_loc, a, b)
    
    def __repr__(self):
        repr = "(Cropping L-Channel Image,\n"
        repr += f"  Crop size = {self.crop_size}\n"
        repr += ")"
        return repr
    
    def crop_img(self, im_l, hintloc, a, b, fixmask=False):
        C,H,W = im_l.shape
        assert H==W, \
            f'The input of Corped patch module is transfomred image. Input images have the size with H:{H} W:{W}'
        if len(hintloc)==0:
    
            dummy_len = self.default_hint_len
            dummy_img, dummy_mask= \
                torch.zeros(dummy_len,3,self.P,self.P), torch.zeros(self.token_num, self.default_hint_len) 
            return dummy_img, dummy_mask
        
        im_a, im_b= torch.ones_like(im_l)*a, torch.ones_like(im_l)*b
        hint_img = torch.cat([im_l,im_a,im_b], dim=0)
        cimg = [torch.zeros(1,3,self.P,self.P).float()]*len(hintloc)

        RT = int(self.token_num**0.5)
        mask = torch.zeros(RT,RT, 1)
        
        center = int(np.sqrt(self.hint_size))
         
        for id, loc in enumerate(hintloc):

            x, y = loc
            xmin = x - self.crop_size
            xmax = x + self.crop_size
            ymin = y - self.crop_size
            ymax = y + self.crop_size

            cropped_img = pad_and_crop_image(hint_img, x, y, pad_size=self.crop_size)
            _, _H, _W = cropped_img.shape

            # center: always crop the image 
            # if self.center: 
            _mask = torch.ones_like(cropped_img[0,:]) #무조건 16 * 16
            if self.hint_size ==1:
                _mask[_H//2,_W//2]=0 # 
            else:
                _mask[_H//2-center:_H//2+center, _W//2-center:_W//2+center]=0
            # else:
            #     _mask = torch.ones_like(hint_img[0])
            #     _mask[y:y+self.hint_size,x:x+self.hint_size] = 0 # ignore coner case (hint loc: end of rx, ry)
            #     _mask = _mask[ymin:ymax, xmin:xmax]
            
            # if self.avg_hint:

            _cropped_img = F.interpolate(cropped_img.unsqueeze(0), size=(_H // self.hint_size, _W // self.hint_size), mode='bilinear',antialias=True)
            _mask = F.interpolate(_mask.unsqueeze(0).unsqueeze(0), size=(_H // self.hint_size, _W // self.hint_size), mode='nearest').bool()
            _cropped_img[:, 1, :, :].masked_fill_(_mask.squeeze(1), 0)
            _cropped_img[:, 2, :, :].masked_fill_(_mask.squeeze(1), 0) 
            x_ab = F.interpolate(_cropped_img, scale_factor=self.hint_size, mode='bilinear', antialias=True)[:, 1:, :, :]
            # TODO 
            cropped_img = torch.cat([cropped_img[0, :, :].unsqueeze(0), x_ab[0]], dim=0)
                       
            cropped_img = F.interpolate(cropped_img.unsqueeze(0), size = (self.P,self.P)) # 3HW, lab image          

            cimg[id]  = cropped_img
            # x+ y*16 
            ymin, xmin, ymax, xmax = max(ymin,0)//self.P, max(xmin,0)//self.P, min(ymax,self.input_size)//self.P, min(xmax,self.input_size)//self.P             
            mask[ymin:ymax+1,xmin:xmax+1,id] = 1

        mask = mask.view(RT*RT,-1)
        cimg= torch.cat(cimg,dim=0)
        
        return cimg, mask
    

if __name__ == '__main__':
    l_img = torch.rand(1,224,224) 
    x, y = 100, 100
    a, b = 0.1, 0.
    lc, rc = (x-7,y-7), (x+8,y+8) #default
    interactive_sampler(l_img, [x, y], lc, rc, a, b)
    # hint_mask = torch.zeros(