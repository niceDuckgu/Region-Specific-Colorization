# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import os
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torchvision

import numpy as np

import utils
from losses import HuberLoss

from einops import rearrange
from timm.models import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import lab2rgb, psnr, rgb2lab, seed_worker
from utils import viz_fixed_val_dataloder

import modeling.modeling as modeling
import modeling.register_model 
from datasets import build_fixed_validation_dataset_cropped_hintlab_fixmask
from options.infer_options import get_args

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        max_hint_len=args.max_hint_len,
    )
    return model


def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    psnr_sum = dict(zip(args.val_hint_list, [0.] * len(args.val_hint_list)))
    total_shown = 0

    args.hint_dirs = [osp.join(args.val_hint_dir, f'h{args.val_hint_size}:{i}') for i in args.val_hint_list]
    dataset_val = build_fixed_validation_dataset_cropped_hintlab_fixmask(args)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=seed_worker,
        shuffle=False,
    )

    with torch.no_grad():
        pbar = tqdm(desc=f'Evaluate', ncols=100, total=len(data_loader_val) * len(args.val_hint_list))
        for step, batch in enumerate(data_loader_val):     
            images = batch['image']
            bool_hints = batch['mask'] # B*val_hint_list*H*W
            cropped_hint_l = batch['cropped_hint'] # B* val_hint_list*H*W
            hint_mask = batch['hint_mask']
            names = batch['name']
            images = images.to(device, non_blocking=True)
            cropped_hint_l = cropped_hint_l.to(device, non_blocking=True)
            hint_mask= hint_mask.to(device, non_blocking=True)
            B, _, H, W = images.shape
            h, w = H // patch_size[0], W // patch_size[1]

            # batch preparation
            images = images.to(device, non_blocking=True)
            images_lab = rgb2lab(images, 50, 100, 110)
            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=patch_size[0], p2=patch_size[1])
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
            images_l =images_lab[:,:1,:,:] 

            for idx, count in enumerate(args.val_hint_list):
                bool_hint = bool_hints[:, idx]
                bool_hint = bool_hint.to(device, non_blocking=True).flatten(1).to(torch.bool)
                bool_hint = ~bool_hint
                with torch.cuda.amp.autocast():
                    outputs = model(images_l.clone(), bool_hint.clone() ,cropped_hint_l[:, idx].clone(), hint_mask[:, idx].clone())
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
                    
                    # loss = loss_func(input=outputs, target=labels[:, :, :, 1:])

                # val
                # loss_value = loss.item()
                # metric_logger.update(loss=loss_value)

                pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
                pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                          h=h, w=w, p1=patch_size[0], p2=patch_size[1])
                pred_imgs = lab2rgb(pred_imgs_lab)

                psnr_sum[count] += psnr(images, pred_imgs).item() * B

                if args.pred_dir is not None:
                    img_save_dir = osp.join(args.pred_dir, f'h{args.hint_size}:n{count}')
                    os.makedirs(img_save_dir, exist_ok=True)
                    for name, pred_img in zip(names, pred_imgs):
                        torchvision.utils.save_image(pred_img.unsqueeze(0), osp.join(
                            img_save_dir, osp.splitext(name)[0] + '.png'))
                pbar.update()

            total_shown += B
            # pbar.set_postfix({'psnr@10': psnr_sum.get(10) / total_shown})
        pbar.close()

    # print("Averaged stats:", metric_logger)
    print(f'Total shown: {total_shown}')
    for key in psnr_sum.keys():
        print(f'PSNR {key}: {psnr_sum[key]/total_shown}')



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    main(args)

