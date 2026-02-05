import os
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize, Resize, Pad

from dataset_folder import ImageFolder, ImageWithFixedHint, ImageWithFixedHintAndCoord
from hint_generator import InteractiveHintGenerator, RandomHintGenerator, EdgeRandomHintGenerator, MixEdgeUniformGenerator

from utils import rgb2lab, lab2rgb

class DataTransformationFixedHintLABFixMask:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.hint_size = args.hint_size
        self.img_transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
        ])
        self.crop_hint_generator = CroppedImageGeneratorFromLABHintFixMask(args.P, args.input_size, args.val_crop_ratio, 
                                                                 args.max_hint_len, hint_size=args.hint_size)
        hint_dirs = args.hint_dirs
        if isinstance(args.hint_dirs, str):
            hint_dirs = [args.hint_dirs]
        self.num_hint = [int(os.path.basename(hint_dir).split(':')[-1])
                         for hint_dir in hint_dirs]  # hint subdir should be formed h#-n##

    def __call__(self, img, hint_coords):

        image, hint = self.img_transform(img), self.coord2hint(hint_coords)
        cropped_img, crop_ratio, hint_loc, hint_mask = [], [], [], []
        
        for idx in range(hint.shape[0]):
            hints = hint[idx,:,:].view(-1)
            _cropped_img, _crop_ratio, _hint_loc, _hint_mask= self.crop_hint_generator(image, hints)
            cropped_img.append(_cropped_img)
            crop_ratio.append(_crop_ratio)
            hint_loc.append(_hint_loc)
            hint_mask.append(_hint_mask)

        return {'image' : image, 'mask' : hint, 'cropped_hint' : torch.stack(cropped_img, axis=0), 
                'crop_ratio': torch.stack(crop_ratio, axis=0), 
                'hint_loc': torch.stack(hint_loc, axis=0), 
                'hint_mask': torch.stack(hint_mask, axis=0)}

    def coord2hint(self, hint_coords):
        hint = torch.ones((len(hint_coords), self.input_size // self.hint_size, self.input_size // self.hint_size))
        for idx, hint_coord in enumerate(hint_coords):
            for y, x in hint_coord:
                hint[idx, y // self.hint_size, x // self.hint_size] = 0
        return hint

    def __repr__(self):
        repr = "(DataTransformationFixedHint,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr

class DataTransformationForHintColorizationLABFixMask:
    def __init__(self, args):
        self.transform = Compose([
            Resize((args.input_size, args.input_size)),
            ToTensor(),
        ])

        if args.hint_generator == 'RandomHintGenerator':
            self.hint_generator = RandomHintGenerator(args.input_size, args.hint_size, args.num_hint_range)
        elif args.hint_generator == 'InteractiveHintGenerator':
            self.hint_generator = InteractiveHintGenerator(args.input_size, args.hint_size)
        elif args.hint_generator == 'EdgeRandomHintGenerator':
            self.hint_generator = EdgeRandomHintGenerator(args.input_size, args.hint_size)
        elif args.hint_generator == 'MixEdgeUniformGenerator':
            self.hint_generator = MixEdgeUniformGenerator(args.input_size, args.hint_size)
        else:
            raise NotImplementedError(f'{args.hint_generator} is not exist.')
         
        self.crop_hint_generator = CroppedImageGeneratorFromLABHintFixMask(args.P, args.input_size, args.crop_ratio, 
                                                                 args.max_hint_len, args.avg_hint, args.hint_size)
        self.patch_size = args.P
        self.ratio = args.crop_ratio 

    def __call__(self, image):
        # hint generator ==> if edge: th, image is input image, th=0.8
 
        image = self.transform(image)
        if 'Edge' in self.hint_generator.__class__.__name__:
            hint = self.hint_generator(image=image, th=0.5)
        else: 
            hint = self.hint_generator()  

        # The scale of the image must be 0~1
        cropped_img, crop_ratio, hint_loc, hint_mask= self.crop_hint_generator(image, hint) #H*P*P
        # return the mask tensor s.t (H*W) size, token idx ==1 or ratio (make the mode: ratio mode or binary mode) 
        # Sizes of the ouputs: c*h*w, h*w, hint*patch*patch, hint*2, Tokennum *(Maxhint)
        return {'image':image, 'mask':hint, 'cropped_hint':cropped_img, 
                'crop_ratio': crop_ratio, 'hint_loc': hint_loc, 'hint_mask': hint_mask}
    
    def __repr__(self):
        repr = "(DataTransformationForHintColorization,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Hint generator = %s,\n" % str(self.hint_generator)
        repr += "  Cropped Hint generator = %s,\n" % str(self.crop_hint_generator)
        repr += ")"
        return repr

class CroppedImageGeneratorFromLABHintFixMask:
    # iColoriT manner hint encoder 
    def __init__(self, P, input_size, crop_ratio, max_hint_len, avg_hint=True, hint_size=2):
        self.P = P
        self.input_size = input_size
        self.ratio = crop_ratio 
        self.max_hint_len = max_hint_len+1 # null_hint_masking
        self.token_num = (input_size//P)**2
        self.avg_hint= avg_hint
        self.hint_size = hint_size

    def __call__(self, image, hint, **kwargs):
        return self.crop_img(image, hint)
    
    def __repr__(self):
        repr = "(Cropping L-Channel Image,\n"
        repr += f"  Crop ratio = {self.ratio}\n"
        repr += ")"
        return repr
    
    def get_hint_loc(self, hint, hint_size=2):
        # hint shape L: H*W//hint_size**2 : hint loc
        coor = np.where(hint==0)
        self.num_hint = len(coor[0])
        x_coor = coor%np.array(self.input_size//hint_size) 
        y_coor = coor//np.array(self.input_size//hint_size)
        return [[x*hint_size,y*hint_size] for x, y in zip(x_coor[0],y_coor[0])]
   
    # 230825 Null token hint masking 
    # IMPORTANT !
    def nulltoken_masking(self, mask):
        # index of null token=-1 ==> h h h, ..., m, m , ... , null toke 
        _mask = torch.sum(mask, dim=-1)==0
        _mask_ = mask.clone()
        _mask_= _mask_[_mask]
        _mask_[:,-1] = 1
        mask[_mask] = _mask_
        return mask    
    
    def crop_img(self, image, hint):
        '''
        input: image with c*h*w, hint with (h*w) size
        output: cropped and resized L-channel pathes & randomly selected ratio r
        '''
        C,H,W = image.shape
        assert H==W, \
            f'The input of Corped patch module is transfomred image. Input images have the size with H:{H} W:{W}'
        hintloc = self.get_hint_loc(hint, self.hint_size)
        dummy_len = self.max_hint_len-self.num_hint 
        dummy_img, dummy_ratio, dummy_loc, dummy_mask= \
            torch.zeros(dummy_len,3,self.P,self.P),torch.zeros(dummy_len,2), \
            torch.ones(dummy_len)*(-1), torch.zeros(self.token_num, self.max_hint_len) 

        if self.num_hint ==0:
            return dummy_img, dummy_ratio, dummy_loc, dummy_mask
        image = rgb2lab(image.unsqueeze(0))[0,:]

        cimg, ratio, hint_loc = [torch.zeros(1,3,self.P,self.P).float()]*len(hintloc), [torch.zeros(2)]*len(hintloc), [0]*len(hintloc)
        padding = int(4 * self.P) 
        
        pad = Pad(padding, 0, padding_mode='constant') 

        image = pad(image)

        RT = int(self.token_num**0.5)
        mask = torch.zeros(RT,RT, self.max_hint_len)
        
        center = int(np.sqrt(self.hint_size))
        for id, loc in enumerate(hintloc):

            rx, ry = 1, 1 
            x, y = loc
            ymin, ymax, xmin, xmax = y-int(self.P//2*ry), y+int(self.P//2*ry), x-int(self.P//2*rx), x+int(self.P//2*rx)
            cropped_img = image[:,ymin+padding+center:ymax+padding+center,
                                 xmin+padding+center:xmax+padding+center]
            # fm = full_mask[0,0,ymin+padding+center:ymax+padding+center, xmin+padding+center:xmax+padding+center] 
            _, _H, _W = cropped_img.shape
            _mask = torch.ones_like(cropped_img[0,:])
            if self.hint_size ==1:
                _mask[_H//2,_W//2]=0 #
            else:
                _mask[_H//2-center:_H//2+center, _W//2-center:_W//2+center]=0
            if self.avg_hint:                
                _cropped_img = F.interpolate(cropped_img.unsqueeze(0), size=(_H // self.hint_size, _W // self.hint_size), mode='bilinear',antialias=True)
                _mask = F.interpolate(_mask.unsqueeze(0).unsqueeze(0), size=(_H // self.hint_size, _W // self.hint_size), mode='nearest').bool()
                _cropped_img[:, 1, :, :].masked_fill_(_mask.squeeze(1), 0)
                _cropped_img[:, 2, :, :].masked_fill_(_mask.squeeze(1), 0)
                x_ab = F.interpolate(_cropped_img, scale_factor=self.hint_size, mode='bilinear', antialias=True)[:, 1:, :, :]
                cropped_img = torch.cat([cropped_img[0, :, :].unsqueeze(0), x_ab[0]], dim=0)

            else:
                _mask = _mask.bool()
                cropped_img[1, :, :].masked_fill_(_mask.squeeze(), 0)
                cropped_img[2, :, :].masked_fill_(_mask.squeeze(), 0)
            
            cropped_img = F.interpolate(cropped_img.unsqueeze(0), size = (self.P,self.P)) # 3*H*W, lab image          

            # resampling for mask attention 
            rx, ry = np.random.choice(self.ratio, 2)
            cimg[id], ratio[id], hint_loc[id] = cropped_img, torch.Tensor([[rx,ry]]), torch.Tensor([x+y*H])
            
            # x+ y*16 
            ymin, ymax, xmin, xmax = y-int(self.P//2*ry), y+int(self.P//2*ry), x-int(self.P//2*rx), x+int(self.P//2*rx)
            ymin, xmin, ymax, xmax = max(ymin,0)//self.P, max(xmin,0)//self.P, min(ymax,self.input_size)//self.P, min(xmax,self.input_size)//self.P             
            mask[ymin:ymax+1,xmin:xmax+1,id] = 1
        mask = mask.view(RT*RT,-1)
        mask = self.nulltoken_masking(mask=mask)
        cimg= torch.cat(cimg,dim=0)

        hint_loc = torch.Tensor(hint_loc)
        ratio = torch.cat(ratio,dim=0)

        return torch.cat([cimg, dummy_img],dim=0), torch.cat([ratio, dummy_ratio],dim=0), \
            torch.cat([hint_loc, dummy_loc], dim=0), mask



# 08 12 Interactive loader 

class FixedCroppedImageGeneratorFromLABHint:
    # iColoriT manner hint encoder 
    def __init__(self, P, input_size, hint_size=2 , center=True):
        # center: if True: click points are center points of the rx, ry 
        #         else: click points and rx, ry are i.d
        self.P = P
        self.input_size = input_size
        self.token_num = (input_size//P)**2
        self.hint_size = hint_size
        self.center = center

    def __call__(self, image, x, y, rx, ry, a, b):
        return self.crop_img(image, x, y, rx, ry, a, b)
    
    def __repr__(self):
        repr = "(Cropping L-Channel Image,\n"
        repr += f"  Crop ratio = {self.ratio}\n"
        repr += ")"
        return repr

    def crop_img(self, image, x, y, lc, rc, a, b):
        C,H,W = image.shape
        assert H==W, \
            f'The input of Corped patch module is transfomred image. Input images have the size with H:{H} W:{W}'
        hintloc = [[x,y]]
        image = rgb2lab(image.unsqueeze(0))[0,:]

        cimg = [torch.zeros(1,3,self.P,self.P).float()]*len(hintloc)

        RT = int(self.token_num**0.5)
        mask = torch.zeros(RT,RT, 1)
        
        center = int(np.sqrt(self.hint_size))

        for id, loc in enumerate(hintloc):
            x, y = loc
            ymin, ymax, xmin, xmax = lc[1], rc[1], lc[0], rc[0]
            cropped_img = image[:,ymin:ymax, xmin:xmax]
            _, _H, _W = cropped_img.shape

            if self.center: 
                _mask = torch.ones_like(cropped_img[0,:])
                if self.hint_size ==1:
                    _mask[_H//2,_W//2]=0 # 
                else:
                    _mask[_H//2-center:_H//2+center, _W//2-center:_W//2+center]=0
            else:
                _mask = torch.ones_like(image[0:])
                _mask[y:y+self.hint_size,x:x+self.hint_size] = 0 # ignore coner case (hint loc: end of rx, ry)
                _mask = _mask[ymin:ymax, xmin:xmax]
            # if self.avg_hint:                
            _cropped_img = F.interpolate(cropped_img.unsqueeze(0), size=(_H // self.hint_size, _W // self.hint_size), mode='bilinear',antialias=True)
            _mask = F.interpolate(_mask.unsqueeze(0).unsqueeze(0), size=(_H // self.hint_size, _W // self.hint_size), mode='nearest').bool()
            _cropped_img[:, 1, :, :].masked_fill_(_mask.squeeze(1), 0)
            _cropped_img[:, 2, :, :].masked_fill_(_mask.squeeze(1), 0)
            _cropped_img[:, 1, :, :].masked_fill_(~(_mask.squeeze(1)), a)
            _cropped_img[:, 2, :, :].masked_fill_(~(_mask.squeeze(1)), b)
            x_ab = F.interpolate(_cropped_img, scale_factor=self.hint_size, mode='bilinear', antialias=True)[:, 1:, :, :]

            cropped_img = torch.cat([cropped_img[0, :, :].unsqueeze(0), x_ab[0]], dim=0)
            
            cropped_img = F.interpolate(cropped_img.unsqueeze(0), size = (self.P,self.P)) # 3*H*W, lab image          

            cimg[id]  = cropped_img
            # x+ y*16 
            ymin, xmin, ymax, xmax = max(ymin,0)//self.P, max(xmin,0)//self.P, min(ymax,self.input_size)//self.P, min(xmax,self.input_size)//self.P             
            mask[ymin:ymax+1,xmin:xmax+1,id] = 1

        mask = mask.view(RT*RT,-1)
        cimg= torch.cat(cimg,dim=0)
               
        return cimg, mask



# 0907
def build_pretraining_dataset_cropped_hintlab_fixmask(args, without_tf=False):
    transform = DataTransformationForHintColorizationLABFixMask(args) if not without_tf else None
    print("Data Trans = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)
def build_fixed_validation_dataset_cropped_hintlab_fixmask(args):
    transform = DataTransformationFixedHintLABFixMask(args)
    print("Data Trans = %s" % str(transform))
    return ImageWithFixedHint(args.val_data_path, args.hint_dirs, transform=transform,
                              return_name=args.return_name, gray_file_list_txt=args.gray_file_list_txt)
