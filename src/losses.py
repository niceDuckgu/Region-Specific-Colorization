import torch.nn as nn
import torch


#0916 add loss for num of the attention layers
class PatchAwareLoss(nn.Module):
    def __init__(self, delta=.01):
        super(PatchAwareLoss, self).__init__()
        self.delta = delta 
    
    def __call__(self, input, target, hint_mask):
        B, T, H  = hint_mask.shape 
        hint_mask = hint_mask-1.
        hint_mask= torch.sum(hint_mask, dim=-1)
        weight = hint_mask/(hint_mask.sum(dim=-1).unsqueeze(-1)+1e6)*T #B*T
        mask = torch.zeros_like(input)
        mann = torch.abs(input - target)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl * mask + self.delta * (mann - .5 * self.delta) * (1 - mask)
        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        loss = weight.unsqueeze(-1).unsqueeze(-1)*loss
        return torch.sum(loss, dim=-1, keepdim=False).mean()


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, input, target):
        mask = torch.zeros_like(input)
        mann = torch.abs(input - target)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl * mask + self.delta * (mann - .5 * self.delta) * (1 - mask)
        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=-1, keepdim=False).mean()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, input, target):
        return torch.sum(torch.abs(input - target), dim=-1, keepdim=False).mean()


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, input, target):
        return torch.sum((input - target)**2, dim=-1, keepdim=False).mean()

if __name__ =='__main__':
    labels = torch.randn(3,196,256,2)
    inputs = torch.ones(3,196,256,2)
    hint_masks = torch.ones(3,196,100)
    losses = PatchAwareLoss()
    losses(labels, inputs, hint_masks)