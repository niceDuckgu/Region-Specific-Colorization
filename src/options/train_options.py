import os
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser('iColoriT training scripts', add_help=False)
    # Training
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--save_args_pkl', action='store_true', help='Save args as pickle file')
    parser.add_argument('--no_save_args_pkl', action='store_false', dest='save_args_pkl', help='')
    parser.set_defaults(save_args_pkl=True)
    parser.add_argument('--save_args_txt', action='store_true', help='Save args as txt file')

    # Dataset
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--data_path', default='data/train/images', type=str, help='dataset path')
    parser.add_argument('--val_data_path', default='data/val/images', type=str, help='validation dataset path')
    parser.add_argument('--val_hint_dir', type=str, help='hint directory for fixed validation', default='./data/hint')
    parser.add_argument('--output_dir', default='outputs', help='relative path where checkpoints are saved')
    parser.add_argument('--log_dir', default='runs', help='relative path for tensorboard logs')
    parser.add_argument('--resume', default='', help='path of checkpoint directory (force_resume should be True)')
    parser.add_argument('--force_resume', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch (changed by resume function if needed)')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--gray_file_list_txt', type=str, default='', help='use gray file list to exclude them')
    parser.add_argument('--return_name', action='store_true', help='return name for saving (False for train)')

    parser.add_argument('--crop_ratio', default=[0.25,0.5,1,2,4], type=list, help='Cropped patch resizing ratio')
    # parser.add_argument('--crop_ratio', default=[1], type=list, help='Cropped patch resizing ratio')
    
    parser.add_argument('--P', default=16, type=int, help='Patch size')
    parser.add_argument('--val_crop_ratio', default=[1], type=list, help='Cropped patch resizing ratio')

    # Model
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to train')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode', choices=['linear', 'cnn', 'locattn'])
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--mask_mode', default='none', 
                        choices = ['none', 'naive', 'cent', 'density',
                                    'ddpm_emb','ddpm_emb_per_point',
                                    'ddpm_emb_per_point_channel',
                                    'ddpm_emb_per_point_channel_woloc'],
                        type=str)

    # Hint Generator
    parser.add_argument('--hint_generator', type=str, default='RandomHintGenerator')
    parser.add_argument('--num_hint_range', default=[0, 150], type=int, nargs=2, help='# hints range for each image')
    parser.add_argument('--hint_size', default=1, type=int, help='size of the hint region is given by (h, h)')
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--val_hint_list', default=[1, 10, 100], nargs='+')
    parser.add_argument('--max_hint_len', default=150, type=int)
    # loss

    parser.add_argument('--patch_loss', action='store_true',help='Patch aware loss')
    parser.add_argument('--beta', type=float, default=0.1, help='The weight of Patch aware loss')
    parser.set_defaults(patch_loss=False)

    # Learning rate scheduling
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, help='steps to warmup LR, priority: steps -> epochs')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    # distributed training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    # debug parameter
    parser.add_argument('--model_path', type=str, help='checkpoint path of model', default='checkpoint.pth')
    return parser.parse_args()


