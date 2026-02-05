import os
import time
import pickle
import argparse
import os.path as osp

from glob import glob

def get_args():
    parser = argparse.ArgumentParser('Infer Colorization', add_help=False)
    # For evaluation
    parser.add_argument('--model_path', type=str, help='checkpoint path of model', default='checkpoint.pth')
    parser.add_argument('--model_args_path', type=str, help='args.pkl path of model', default='')
    parser.add_argument('--val_data_path', default='data/val/images', type=str, help='validation dataset path')
    parser.add_argument('--val_hint_dir', type=str, help='hint directory for fixed validation', default='data/hint')
    parser.add_argument('--pred_dir', type=str, default='predictions', help='relative folder to store predictions')
    parser.add_argument('--gray_file_list_txt', type=str, default='', help='use gray file list to exclude them')
    parser.add_argument('--return_name', action='store_true', help='return name for saving (True for test)')
    parser.add_argument('--no_return_name', action='store_false', dest='return_name', help='')
    parser.set_defaults(return_name=True)

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--P', default=16, type=int, help='Patch size')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.add_argument('--crop_ratio', default=[0.25,0.5,1,2,4], type=list, help='Cropped patch resizing ratio')

    # Hint Loader
    parser.add_argument('--num_hint_range', default=[0, 150], type=int, nargs=2, help='# hints range for each image')
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    # parser.set_defaults(avg_hint=True)

    # If occure the out of memory, set the pin mem = False 
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to inference')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode', choices=['linear', 'cnn', 'locattn'])
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Hint generator parameter
    parser.add_argument('--hint_size', default=2, type=int, help='size of the hint region is given by (h, h)')
    parser.add_argument('--val_hint_size', default=2, type=int, help='validationsize of the hint region is given by (h, h)')
    parser.add_argument('--max_hint_len', default=200, type=int)
    parser.add_argument('--val_crop_ratio', default=[1], type=float, nargs='+',help='Cropped patch resizing ratio')
    parser.add_argument('--val_hint_list', default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500], nargs='+')
    # parser.add_argument('--val_hint_list', default=[500], nargs='+')

    # SAVE PATH

    args = parser.parse_args()

    if osp.isdir(args.model_path):
        all_checkpoints = glob(osp.join(args.model_path, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.model_path = os.path.join(args.model_path, f'checkpoint-{latest_ckpt}.pth')
    print(f'Load checkpoint: {args.model_path}')

    if args.model_args_path:
        with open(args.model_args_path, 'rb') as f:
            train_args = vars(pickle.load(f))
            model_keys = ['model', 'use_rpb', 'head_mode', 'drop_path', 'mask_mode']
            for key in model_keys:
                if key in train_args.keys():
                    setattr(args, key, train_args[key])
                else:
                    print(f'{key} is not in {args.model_args_path}. Please check the args.pkl')
            time.sleep(3)
    print(f'Load args: {args.model_args_path}')
    # args.val_hint_list = [int(h) for h in args.val_hint_list]
    args.val_hint_list = [0,1,2,5,10,20,50,100,200, 500]
    # for count in args.val_hint_list:
    #     os.makedirs(osp.join(args.pred_dir, f'h{args.hint_size}-n{count}'), exist_ok=True)

    return args
