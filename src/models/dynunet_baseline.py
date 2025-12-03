import torch.nn as nn
from monai.networks.nets import DynUNet

class DynUNetWrapper(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base=32):
        super().__init__()
        filters = [base, base*2, base*4, base*8, base*16]
        kernel = [(3,3,3)] * 5
        strides = [(1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2)]
        upk = [(2,2,2)] * 4

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            upsample_kernel_size=upk,
            res_block=True,
        )
    def forward(self, x):
        return self.model(x), None
