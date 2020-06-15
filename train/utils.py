import torch
import torch.nn as nn
import torch.nn.init as init


def torch_same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    '''Init layer parameters.'''
    torch_same_seeds(10)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        m.weight.data = torch.clamp(m.weight, -1, 1)
        if hasattr(m, "bias"):
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        m.weight.data = torch.clamp(m.weight, -1, 1)
        if hasattr(m, "bias"):
            init.constant_(m.bias, 0)
