import sys
sys.path.insert(0,'../')
from adjusted_VarNet.fft_utils import torch_abs
from general_utils import *


class Concatenated_Networks_kernels(nn.Module):
    def __init__(self, first_net, sec_net):
        super().__init__()
        self.varnet = first_net
        self.segm_unet = sec_net
        self.kernel = []
    def forward(self, x):
        x_var, self.kernel = self.varnet(x)
        if len(x_var.size())==3:  # NxHxW
            x = torch_abs(x_var).unsqueeze(1)
        else:  # NxFxHxW
            x = torch_abs(x_var)
            x = self.segm_unet(x)
        return x, x_var
    def get_varnet_reference(self, ref):
        return self.varnet.get_reference(ref, self.kernel)

    def save_state_dict(self, save_dir, best=True):
        if best:
            torch.save(self.varnet.state_dict(), str(save_dir / 'varnet_best.h5'))
            torch.save(self.segm_unet.state_dict(), str(save_dir / 'nnunet_best.h5'))
        else:
            torch.save(self.varnet.state_dict(), str(save_dir / 'varnet_last.h5'))
            torch.save(self.segm_unet.state_dict(), str(save_dir / 'nnunet_last.h5'))

