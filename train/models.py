import torch
import torch.nn as nn


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


class Conv2dAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                 stride=1, padding=1, groups=1,
                 activation=nn.Sigmoid):
        padding = (kernel_size - 1) // 2 if not padding else padding
        super(Conv2dAct, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups),
            activation()
        )


class ConvNet(nn.Module):
    """Pytorch implementation of CNN"""
    def __init__(self, conv_channels=[32, 64],
    			 image_shape=(1, 28, 28), num_classes=10,
                 kernel_size=4, stride=2, padding=1, use_swish=False):
        super(ConvNet, self).__init__()

        # Default input channel is 1
        num_conv = len(conv_channels)
        conv_channels = [image_shape[0]] + conv_channels
        activation = nn.Sigmoid if not use_swish else Swish

        # Build encoder layers
        features = [
            Conv2dAct(conv_channels[i], conv_channels[i+1],
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      activation=activation)
            for i in range(num_conv)]

        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], num_classes))


    def forward(self, x):
        x = self.features(x)
        output = x.view(x.size(0), -1)
        output = self.classifier(output)
        return output

