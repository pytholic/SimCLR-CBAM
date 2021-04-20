import torch
import torch.nn as nn

class custom_unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(custom_unet, self).__init__()

        self.downsample = nn.Sequential(

        )