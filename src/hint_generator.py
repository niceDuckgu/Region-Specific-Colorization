import copy

import torch
import numbers
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import ImageFilter
from skimage.filters import sobel
from torch.nn.functional import softmax
from torchvision.transforms.functional import to_pil_image, to_tensor


######################### Dataset Utils #################################
class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))

def uniform_gen(hintrange, num_hint_location):
    num_hint = np.random.random_integers(hintrange[0], hintrange[1])
    hint = np.hstack([
        np.ones(num_hint_location - num_hint),
        np.zeros(num_hint),
    ])
    np.random.shuffle(hint)
    return hint

def edge_gen(image, num_hint_range, H, W, hint_size, 
             smoothing, gaussian=GaussianSmoothing):
    
    #TODO unifying to numpy 
    C, H, W = image.shape 
    image = to_pil_image(image)
    image = image.resize((H//hint_size, W//hint_size))
    image = to_tensor(gaussian(smoothing)(image))
    
    gray_image = torch.sum(image, dim =0)
    edge_map= torch.Tensor(sobel(gray_image.numpy()))
    edge_dist = softmax(edge_map.view(-1))#.view(C,H,W)
    num_hint = np.random.random_integers(num_hint_range[0], num_hint_range[1])
    index = np.random.choice(len(edge_dist), num_hint, p=edge_dist.numpy())
    hint = torch.ones(len(edge_dist))
    hint[index]=0

    return hint

#########################################################################

class RandomHintGenerator:
    '''
    Use RandomHintGenerator in BEiT as random hint generator
    '''

    def __init__(self, input_size, hint_size=2, num_hint_range=[10, 10]):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_hint_location = self.height * self.width // (hint_size * hint_size)
        self.num_hint_range = num_hint_range

    def __repr__(self):
        repr_str = (f'Hint: total hint locations {self.num_hint_location},'
                    f'number of hints range {self.num_hint_range}')
        return repr_str

    def __call__(self, **kwargs):
        return uniform_gen(self.num_hint_range, self.num_hint_location)


class InteractiveHintGenerator:
    ''' Interactive hint generator by user input '''

    def __init__(self, input_size, hint_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.hint_size = hint_size

        # set hyper-parameters
        self.height, self.width = input_size
        self.hint_size = hint_size
        self.num_hint_location = self.height * self.width // (hint_size * hint_size)

        self.hint = np.ones((self.height // hint_size, self.width // hint_size))
        self.coord_xs, self.coord_ys = None, None

    def __repr__(self):
        repr_str = f"Hint: total hint locations {self.num_hint_location}"
        return repr_str

    def __call__(self, **kwargs):
        if self.coord_xs is None:
            self.coord_xs, self.coord_ys = [], []
            return copy.deepcopy(self.hint), torch.tensor((self.coord_xs, self.coord_ys)).T
        while True:
            coord_x = float(input('coord_x: '))
            coord_y = float(input('coord_y: '))
            if coord_x >= 0 and coord_y >= 0 and coord_x < self.height and coord_y < self.width:
                break
            print(f'coord_x, coord_y should be in [0, {self.height}) [0, {self.width})')

        self.coord_xs.append(coord_x)
        self.coord_ys.append(coord_y)
        coord_x = int(coord_x // self.hint_size)
        coord_y = int(coord_y // self.hint_size)
        coord_x = self.hint.shape[0] - 1 if coord_x >= self.hint.shape[0] else coord_x
        coord_y = self.hint.shape[1] - 1 if coord_y >= self.hint.shape[1] else coord_y

        self.hint[coord_x, coord_y] = 0

        return copy.deepcopy(self.hint), torch.tensor((self.coord_xs, self.coord_ys)).T

class EdgeRandomHintGenerator:
    '''
    Use RandomHintGenerator in BEiT as random hint generator
    '''

    def __init__(self, input_size, hint_size=2, num_hint_range=[10, 10]):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size 
        self.hint_size = hint_size
        self.num_hint_location = self.height * self.width // (hint_size * hint_size)
        self.num_hint_range = num_hint_range
        self.gaussian = GaussianSmoothing
    def __repr__(self):
        repr_str = (f'Hint: total hint locations {self.num_hint_location},'
                    f'number of hints range {self.num_hint_range}')
        return repr_str

    def __call__(self, image, **kwargs):
        return self.distgen(image)
        
    def distgen(self, image, smoothing = 3):

        return edge_gen(image, self.num_hint_range,self.height, self.width,
                        self.hint_size, smoothing, self.gaussian)
    
class MixEdgeUniformGenerator:

    def __init__(self, input_size, hint_size=2, num_hint_range=[10, 10]):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size 
        self.hint_size = hint_size
        self.num_hint_location = self.height * self.width // (hint_size * hint_size)
        self.num_hint_range = num_hint_range
        self.gaussian = GaussianSmoothing
    def __repr__(self):
        repr_str = (f'Hint: total hint locations {self.num_hint_location},'
                    f'number of hints range {self.num_hint_range}')
        return repr_str

    def __call__(self, image, smoothing=3, th=0.5):
         
        return self.distgen(image, smoothing=smoothing, th=th)
        
    def distgen(self, image, smoothing, th):
        rand = torch.rand(1)
        th = 10

        if rand.item()<th/self.num_hint_range[1]: # 1: randn & 0 edge based sampling
            return edge_gen(image, [0, th], self.height, self.width, 
                            self.hint_size, smoothing, self.gaussian)
        
        else:
            return uniform_gen([th, self.num_hint_range[1]], self.num_hint_location)

if __name__ == '__main__':

    from PIL import Image
    from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize, Resize
    from torchvision.utils import save_image
    repo_root = Path(__file__).resolve().parent
    image_path = repo_root / "assets" / "flower.jpg"
    image = Image.open(image_path).convert('RGB')
    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    transform = Compose([
            RandomResizedCrop(224),
            ToTensor(),
            Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    image = transform(image)
    outputs = EdgeRandomHintGenerator(224)
    outputs(image)