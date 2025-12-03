import torch.nn as nn
from monai.networks.nets import UNet

class ReturnTuple(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
    def forward(self, x):
        return self.core(x), None

def make_unet3d(in_ch=4, out_ch=4, base=32):
    return ReturnTuple(
        UNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(base, base*2, base*4, base*8, base*10),
            strides=(2,2,2,2),
            num_res_units=2,
            norm="INSTANCE",
        )
    )
