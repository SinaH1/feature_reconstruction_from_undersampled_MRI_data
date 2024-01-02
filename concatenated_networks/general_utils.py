import sys
sys.path.insert(0,'../')
import numpy as np
import torch
from UNet.loss_functions.dice_loss import get_tp_fp_fn_tn
import torch.nn.functional as F
import torch.nn as nn

def dice_coefficient(target, predict):
	return 2 * torch.sum(torch.mul(predict, target))/(torch.sum(predict + target))

def get_TP_TN_FP_FN(target, predict):
	tp = torch.sum(target * predict).item()
	tn = torch.sum((target+predict == 0).float()).item()
	fp = torch.sum((predict - target == 1).float()).item()
	fn = torch.sum((target - predict == 1).float()).item()
	return tp,tn,fp,fn

def dice_coefficient_np(seg_true, seg):
    return 2*np.sum(seg_true * seg)/(2*np.sum(seg_true * seg) + np.sum(np.abs(seg_true-seg)))

def dice_coefficient_unet(seg_true, seg):
    tp, fp, fn, tn = get_tp_fp_fn_tn(seg, seg_true)
    return 2*tp / (2 * tp + fp + fn)

def mse_normed_kernel(net_output, target):
	return torch.sum(torch.square(net_output-target)) / torch.sum(torch.square(target))

class MSELoss(nn.Module):
    def __init__(self, function='standard'): # , batch_dice=False, do_bg=True, smooth=1.):
        """
        function: standard, torch
        """
        super(MSELoss, self).__init__()

        self.function = function

    def forward(self, net_output, target):
        '''
        Parameters
        ----------
        net_output : torch tensor, shape BxFxHxW
        target : torch tensor, shape BxFxHxW

        Returns MSE Loss
        -------
        None.

        '''
        if self.function=='standard':
            loss = torch.sum(torch.square(net_output-target)) / net_output.numel()
        elif self.function == 'torch':
            loss = F.mse_loss(net_output, target)
        elif self.function =='normed_kernel':
            loss = torch.sum(torch.square(net_output-target)) / torch.sum(torch.square(target))
        else:
            raise NotImplementedError("MSE Loss Function needs to be chosen.")
        return loss