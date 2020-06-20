import math
import torch
import torch.nn as nn
from torchvision.models import resnet18
# import torch.nn.init as init


def torch_same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SwishBackend(torch.autograd.Function):
    """Autograd implementation of Swish activation"""
    @staticmethod
    def forward(ctx, input_):
        """Forward pass

        Compute the swish activation and save the input tensor for backward
        """
        output = input_ * torch.sigmoid(input_)
        ctx.save_for_backward(input_)
        return output

    @staticmethod
    def backward(ctx, grade_output):
        """Backward pass

        Compute the gradient of Swish activation w.r.t. grade_ouput
        """
        input_ = ctx.saved_variables[0]
        i_sigmoid = torch.sigmoid(input_)
        return grade_output * (i_sigmoid * (1 + input_ * (1 - i_sigmoid)))


class Swish(nn.Module):
    """ Wrapper for Swish activation function.

    Refs:
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    def forward(self, x):
        # return x * F.sigmoid(x)
        return SwishBackend.apply(x)


# def init_weights(m):
#     '''Init layer parameters.'''
#     torch_same_seeds(10)
#     norm = 0.05
#     if isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight, mode='fan_out')
#         m.weight.data *= norm
#         # m.weight.data = torch.clamp(m.weight, -norm, norm)
#         if hasattr(m, "bias") and m.bias is not None:
#             m.bias.data.uniform_(-norm, norm)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         init.normal_(m.weight, std=1)
#         m.weight.data *= norm
#         # m.weight.data = torch.clamp(m.weight, -norm, norm)
#         if hasattr(m, "bias") and m.bias is not None:
#             m.bias.data.uniform_(-norm, norm)


def init_weights(m):
    norm = 0.1
    # best for L4D2 is 0.1
    if hasattr(m, "weight"):
        torch_same_seeds(10)
        m.weight.data.uniform_(-norm, norm)
        # m.weight.data.normal_(0, 1e-3)
    if hasattr(m, "bias"):
        torch_same_seeds(10)
        # m.weight.data.normal_(0, 1e-3)
        m.bias.data.uniform_(-norm, norm)


class Conv2dAct(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 activation=nn.Sigmoid):
        padding = (kernel_size - 1) // 2 if not padding else padding
        super(Conv2dAct, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups),
            # nn.BatchNorm2d(out_planes),
            activation())


def get_conv_out_size(in_size, num_conv, kernel_size=4, stride=2, padding=1):
    out_size = in_size
    for _ in range(num_conv):
        out_size = math.floor((out_size + 2 * padding - kernel_size) / stride)
        out_size += 1
    return int(out_size)


class ConvNet(nn.Module):
    """Pytorch implementation of CNN"""
    def __init__(self,
                 image_shape=(3, 32, 32),
                 conv_channels=[12, 24, 48],
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 num_classes=10,
                 use_swish=True):
        super(ConvNet, self).__init__()
        # Default input channel is 3
        num_conv = len(conv_channels)
        conv_channels = [image_shape[0]] + conv_channels
        activation = nn.Sigmoid
        # activation = nn.ReLU
        if use_swish:
            activation = Swish
        # Build encoder layers
        features = [
            Conv2dAct(conv_channels[i],
                      conv_channels[i + 1],
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      activation=activation) for i in range(num_conv)
        ]

        # Compute the feature out size
        f_out_size = get_conv_out_size(image_shape[1],
                                       num_conv=num_conv,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding)

        # features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*features)
        # self.classifier = nn.Sequential(
        #     nn.Linear(conv_channels[-1], num_classes))
        out_size = conv_channels[-1] * (f_out_size**2)
        self.classifier = nn.Sequential(nn.Linear(out_size, num_classes))

        # Initialize the model weights
        self.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out


<<<<<<< HEAD
def config_net(net_name="", input_shape=(3, 32, 32), num_classes=10):
    assert net_name in ["CNN_L2D1", "CNN_L2D2",
                        "CNN_L4D1", "CNN_L4D2",
                        "CNN_L4D4", "CNN_L6D2",
                        "CNN_L7D2", "ResNet18"], "{}".format(net_name)

    if net_name == "CNN_L2D1":
        conv_channels = [32, 64]
    elif net_name == "CNN_L2D2":
        conv_channels = [64, 128]
    elif net_name == "CNN_L4D1":
        conv_channels = [32, 64, 128, 256]
    elif net_name == "CNN_L4D2":
        conv_channels = [64, 128, 256, 512]
    elif net_name == "CNN_L4D4":
        conv_channels = [128, 256, 512, 1024]
    elif net_name == "CNN_L6D2":
        conv_channels = [32, 64, 64, 128, 128, 256]
    elif net_name == "CNN_L7D2":
        conv_channels = [64, 64, 64, 128, 128, 256, 256]
    elif net_name == "ResNet18":
        net = config_resnet18(
            input_channels=input_shape[0],
            num_classes=num_classes)
        return net
        # end switch

    net = ConvNet(
        image_shape=input_shape,
        conv_channels=conv_channels,
        num_classes=num_classes
    )
    return net
=======
def config_resnet18(input_channels=1, num_classes=10):
    resnet = resnet18(pretrained=False, progress=True)
    resnet.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False)
    resnet.fc = nn.Linear(512, num_classes)
    # weight initialization
    resnet.apply(init_weights)
    return resnet
>>>>>>> 446fa93db7bb6dce9d906f2862f0821cb47702d2
