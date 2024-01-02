import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from adjusted_VarNet.fft_utils import *
from adjusted_VarNet.misc_utils import *
import copy
from pathlib import Path
from adjusted_VarNet.optimizer import IIPG
from skimage.metrics import structural_similarity

DEFAULT_OPTS = {'kernel_size': 11,
                'features_in': 1,
                'features_out': 24,
                'do_prox_map': True,
                'pad': 11,
                'vmin': -1.0, 'vmax': 1.0,
                'lamb_init': 1.0,
                'num_act_weights': 31,
                'init_type': 'linear',
                'init_scale': 0.04,
                'sampling_pattern': 'cartesian',
                'num_stages': 10,
                'seed': 1,
                'optimizer': 'adam', 'lr': 1e-4,
                'activation': 'rbf',
                'loss_type': 'complex',
                'momentum': 0.,
                'error_scale': 10,
                'loss_weight': 1}


class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """ Forward pass for RBF activation
        Parameters:
        ----------
        ctx:
        input: torch tensor (NxFxCxHxW) 16x48x214x214
            input tensor
        w: torch tensor (1 x C x 1 x 1 x # of RBF kernels)
            weight of the RBF kernels
        mu: torch tensor (# of RBF kernels)
            center of the RBF
        sigma: torch tensor (1)
            std of the RBF
        Returns:
        ----------
        torch tensor: linear weight combination of RBF of input
        """
        num_act_weights = w.shape[-1]
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(num_act_weights):
            tmp = w[:, :, :, :, i] * torch.exp(-torch.square(input - mu[i]) / (2 * sigma ** 2))
            output += tmp
            rbf_grad_input += tmp * (-(input - mu[i])) / (sigma ** 2)
        del tmp
        ctx.save_for_backward(input, w, mu, sigma, rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors
        num_act_weights = w.shape[-1]

        # if ctx.needs_input_grad[0]:
        grad_input = grad_output * rbf_grad_input

        # if ctx.need_input_grad[1]:
        grad_w = w.new_zeros(w.shape)
        for i in range(num_act_weights):
            tmp = (grad_output * torch.exp(-torch.square(input - mu[i]) / (2 * sigma ** 2))).sum((0, 2, 3))
            grad_w[:, :, :, :, i] = tmp.view(w.shape[0:-1])

        return grad_input, grad_w, None, None

class RBFActivation(nn.Module):
    """ RBF activation function with trainable weights """

    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        x_0 = np.linspace(kwargs['vmin'], kwargs['vmax'], kwargs['num_act_weights'], dtype=np.float32)
        mu = np.linspace(kwargs['vmin'], kwargs['vmax'], kwargs['num_act_weights'], dtype=np.float32)
        self.sigma = 2 * kwargs['vmax'] / (kwargs['num_act_weights'] - 1)
        self.sigma = torch.tensor(self.sigma)
        if kwargs['init_type'] == 'linear':
            w_0 = kwargs['init_scale'] * x_0
        elif kwargs['init_type'] == 'tv':
            w_0 = kwargs['init_scale'] * np.sign(x_0)
        elif kwargs['init_type'] == 'relu':
            w_0 = kwargs['init_scale'] * np.maximum(x_0, 0)
        elif kwargs['init_type'] == 'student-t':
            alpha = 100
            w_0 = kwargs['init_scale'] * np.sqrt(alpha) * x_0 / (1 + 0.5 * alpha * x_0 ** 2)
        else:
            raise ValueError("init_type '%s' not defined!" % kwargs['init_type'])
        w_0 = np.reshape(w_0, (1, 1, 1, 1, kwargs['num_act_weights']))
        w_0 = np.repeat(w_0, kwargs['features_out'], 1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.mu = torch.from_numpy(mu)
        self.rbf_act = RBFActivationFunction.apply

    def forward(self, x):
        if not self.mu.device == x.device:
            self.mu = self.mu.to(x.device)
            self.sigma = self.sigma.to(x.device)
        output = self.rbf_act(x, self.w, self.mu, self.sigma)
        return output


class VnMriReconCell(nn.Module):
    """ One cell of variational network """

    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        conv_kernel = np.random.randn(options['features_out'], options['num_features'], options['kernel_size'],
                                      options['kernel_size'], 2).astype(np.float32) \
                      / np.sqrt(options['kernel_size'] ** 2 * 2 * options['num_features'])
        conv_kernel -= np.mean(conv_kernel, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel = torch.from_numpy(conv_kernel)
        if options['do_prox_map']:
            conv_kernel = zero_mean_norm_ball(conv_kernel, axis=(1, 2, 3, 4))

        self.conv_kernel = torch.nn.Parameter(conv_kernel) #, requires_grad=False)

        if self.options['activation'] == 'rbf':
            self.activation = RBFActivation(**options)
        elif self.options['activation'] == 'relu':
            self.activation = torch.nn.ReLU()
        self.lamb = torch.nn.Parameter(torch.tensor(options['lamb_init'], dtype=torch.float32)) #, requires_grad=False)

    def mri_forward_op(self, u, coil_sens, sampling_mask, os=False):
        """
        Forward pass with kspace
        (2X the size)

        Parameters:
        ----------
        u: torch tensor NxHxWx2
            complex input image
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding
        Returns:
        -----------
        kspace of u with applied coil sensitivity and sampling mask
        """
        if os:
            pad_u = torch.tensor((sampling_mask.shape[1] * 0.25 + 1), dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1] * 0.25 - 1), dtype=torch.int16)
            u_pad = F.pad(u, [0, 0, 0, 0, pad_u, pad_l])
        else:
            u_pad = u
        u_pad = u_pad.unsqueeze(2) # NxFx1xHxWx2
        coil_imgs = complex_mul(u_pad, coil_sens)  # NxFxCxHxWx2

        Fu = fftc2d(coil_imgs)  #

        mask = sampling_mask.unsqueeze(1).unsqueeze(-1)   # NxFx1xHxWx1
        mask = mask.repeat([1, 1, 1, 1, 1, 2])  # NxFx1xHxWx2

        kspace = mask * Fu  # NxCxHxWx2
        return kspace

    def mri_adjoint_op(self, f, coil_sens, sampling_mask, os=False):
        """
        Adjoint operation that convert kspace to coil-combined under-sampled image
        by using coil_sens and sampling mask

        Parameters:
        ----------
        f: torch tensor NxCxHxWx2
            multi channel kspace
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding
        Returns:
        -----------
        Undersampled, coil-combined image
        """

        # Apply mask and perform inverse centered Fourier transform
        mask = sampling_mask.unsqueeze(1).unsqueeze(-1)  # NxFx1xHxWx1
        mask = mask.repeat([1, 1, 1, 1, 1, 2])  # NxFx1xHxWx2
        

        Finv = ifftc2d(mask * f)  # NxFxCxHxWx2
        # multiply coil images with sensitivities and sum up over channels
        img = torch.sum(complex_mul(Finv, conj(coil_sens)), 2)

        if os:
            # Padding to remove FE oversampling
            pad_u = torch.tensor((sampling_mask.shape[1] * 0.25 + 1), dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1] * 0.25 - 1), dtype=torch.int16)
            img = img[:, pad_u:-pad_l, :, :]

        return img

    def forward(self, inputs):
        u_t_1 = inputs['u_t']  # NxFxHxWx2
        f = inputs['f']  # NxFxCxHxWx2
        c = inputs['coil_sens']  # NxCxHxW
        m = inputs['sampling_mask']  # NxHxW

        #u_t_1 = u_t_1.unsqueeze(2)  # NxFx1xHxWx2
        # pad the image to avoid problems at the border
        pad = self.options['pad']
        u_t_real = u_t_1[:, :, :, :, 0]
        u_t_imag = u_t_1[:, :, :, :, 1]

        u_t_real = F.pad(u_t_real, [pad, pad, pad, pad], mode='reflect')  # to do: implement symmetric padding
        u_t_imag = F.pad(u_t_imag, [pad, pad, pad, pad], mode='reflect')
        # split the image in real and imaginary part and perform convolution
        u_k_real = F.conv2d(u_t_real, self.conv_kernel[:, :, :, :, 0], stride=1, padding=5)  # NxFx214x214 -> NxF_outx214x214
        u_k_imag = F.conv2d(u_t_imag, self.conv_kernel[:, :, :, :, 1], stride=1, padding=5)  # NxFx214x214 -> NxF_outx214x214
        # add up the convolution results
        u_k = u_k_real + u_k_imag
        # apply activation function
        f_u_k = self.activation(u_k)
        # perform transpose convolution for real and imaginary part
        u_k_T_real = F.conv_transpose2d(f_u_k, self.conv_kernel[:, :, :, :, 0], stride=1, padding=5)
        u_k_T_imag = F.conv_transpose2d(f_u_k, self.conv_kernel[:, :, :, :, 1], stride=1, padding=5)

        # Rebuild complex image
        u_k_T_real = u_k_T_real.unsqueeze(-1)
        u_k_T_imag = u_k_T_imag.unsqueeze(-1)
        u_k_T = torch.cat((u_k_T_real, u_k_T_imag), dim=-1)

        # Remove padding and normalize by number of filter
        Ru = u_k_T[:, :, pad:-pad, pad:-pad, :]  # NxFxHxWx2
        Ru /= self.options['features_out']

        if self.options['sampling_pattern'] == 'cartesian':
            os = False
        elif not 'sampling_pattern' in self.options or self.options['sampling_pattern'] == 'cartesian_with_os':
            os = True

        Au = self.mri_forward_op(u_t_1, c, m, os)
        At_Au_f = self.mri_adjoint_op(Au - f, c, m, os)
        Du = At_Au_f * self.lamb
        u_t = u_t_1 - Ru - Du
        output = {'u_t': u_t, 'f': f, 'coil_sens': c, 'sampling_mask': m, 'kernel': inputs['kernel']}
        return output  # NxHxWx2


class OriginalRecon(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.filter_kernels = None
        
    def forward(self, inputs):
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)
        m = m.unsqueeze(1)

        output = {'u_t': u, 'f': f, 'coil_sens': c, 'sampling_mask': m, 'kernel': self.filter_kernels}
        return output
        

class FeatureExtraction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.num_kernels = self.options['num_features']
        torch.autograd.set_detect_anomaly(True)
        #self.filter_kernels = nn.ParameterList([]).to(options['device'])
        #self.filter_kernels = []
        self.pad_kernels = []
        filter_kernel1 =  torch.tensor(np.array([[0,0,0],[0,1,0],[0,0,0]])).to(options['device']).to(torch.float32)
        filter_kernel2 =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)
        filter_kernel3 =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)
        filter_kernel4 =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)

        self.filter_kernels = nn.ParameterList([nn.Parameter(filter_kernel1, requires_grad=True), nn.Parameter(filter_kernel2, requires_grad=True), nn.Parameter(filter_kernel3, requires_grad=True), nn.Parameter(filter_kernel4, requires_grad=True)])
        self.filter_kernels[0].requires_grad = False
        for i in range(self.num_kernels):
            self.pad_kernels.append(nn.ZeroPad2d((0, int(self.options['image_size'][0] - 3), 0, int(self.options['image_size'][1] - 3))))

    def forward(self, inputs):
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)
        m = m.unsqueeze(1)

        out_u =  []
        out_f = []
        for idx, filt in enumerate(self.filter_kernels):
            # convolution of input image u
            padding = 1
            u_idx = torch.zeros_like(u)
            for i in range(2):
                u_idx[..., i] = F.conv2d(u[..., i], filt.detach().unsqueeze(0).unsqueeze(0), padding=padding)
            out_u.append(u_idx)

            # create modified mri data
            pad = self.pad_kernels[idx]
            fft_kernel = torch.fft.fft2(pad(filt))
            f_complex = fft_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(f.size()[0], f.size()[1],f.size()[2], 1, 1) * torch.complex(f[..., 0], f[..., 1])
            f_idx = torch.zeros_like(f)
            f_idx[..., 0] = torch.real(f_complex)
            f_idx[..., 1] = torch.imag(f_complex)
            
            out_f.append(f_idx)
        
        out_u = torch.cat(out_u, dim=1)
        out_f = torch.cat(out_f, dim=1)
        output = {'u_t': out_u, 'f': out_f, 'coil_sens': c, 'sampling_mask': m, 'kernel': self.filter_kernels}
        return output
    
class FeatureExtraction_1kernel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.num_kernels = self.options['num_features']
        torch.autograd.set_detect_anomaly(True)
        self.pad_kernels = []
        filter_kernel =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)
        self.filter_kernels = nn.ParameterList([nn.Parameter(filter_kernel, requires_grad=True)])

        for i in range(self.num_kernels):
            self.pad_kernels.append(nn.ZeroPad2d((0, int(self.options['image_size'][0] - 3), 0, int(self.options['image_size'][1] - 3))))

    def forward(self, inputs):
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)
        m = m.unsqueeze(1)

        out_u =  []
        out_f = []
        for idx, filt in enumerate(self.filter_kernels):
            # convolution of input image u
            #padding = idx + 1
            padding = 1
            u_idx = torch.zeros_like(u)
            for i in range(2):
                u_idx[..., i] = F.conv2d(u[..., i], filt.detach().unsqueeze(0).unsqueeze(0), padding=padding)
            out_u.append(u_idx)

            # create modified mri data
            pad = self.pad_kernels[idx]
            fft_kernel = torch.fft.fft2(pad(filt))
            f_complex = fft_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(f.size()[0], f.size()[1],f.size()[2], 1, 1) * torch.complex(f[..., 0], f[..., 1])
            f_idx = torch.zeros_like(f)
            f_idx[..., 0] = torch.real(f_complex)
            f_idx[..., 1] = torch.imag(f_complex)
            
            out_f.append(f_idx)
        
        out_u = torch.cat(out_u, dim=1)
        out_f = torch.cat(out_f, dim=1)
        output = {'u_t': out_u, 'f': out_f, 'coil_sens': c, 'sampling_mask': m, 'kernel': self.filter_kernels}
        return output
    

class RandomKernels(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.num_kernels = self.options['num_features']
        torch.autograd.set_detect_anomaly(True)

        self.pad_kernels = []
        filter_kernel1 =  torch.tensor(np.array([[0,0,0],[0,1,0],[0,0,0]])).to(options['device']).to(torch.float32)
        filter_kernel2 =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)
        filter_kernel3 =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)
        filter_kernel4 =  torch.tensor(2*np.random.rand(3,3)-1).to(options['device']).to(torch.float32)
        self.filter_kernels = [nn.Parameter(filter_kernel1, requires_grad=False), nn.Parameter(filter_kernel2, requires_grad=False), nn.Parameter(filter_kernel3, requires_grad=False), nn.Parameter(filter_kernel4, requires_grad=False)]
        self.filter_kernels[0].requires_grad = False
        for i in range(self.num_kernels):
            self.pad_kernels.append(nn.ZeroPad2d((0, int(self.options['image_size'][0] - 3), 0, int(self.options['image_size'][1] - 3))))

    def forward(self, inputs):
        self.pad_kernels = []
        filter_kernel1 =  torch.tensor(np.array([[0,0,0],[0,1,0],[0,0,0]])).to(self.options['device']).to(torch.float32)
        filter_kernel2 =  torch.tensor(2*np.random.rand(3,3)-1).to(self.options['device']).to(torch.float32)
        filter_kernel3 =  torch.tensor(2*np.random.rand(3,3)-1).to(self.options['device']).to(torch.float32)
        filter_kernel4 =  torch.tensor(2*np.random.rand(3,3)-1).to(self.options['device']).to(torch.float32)
        self.filter_kernels = [nn.Parameter(filter_kernel1, requires_grad=False), nn.Parameter(filter_kernel2, requires_grad=False), nn.Parameter(filter_kernel3, requires_grad=False), nn.Parameter(filter_kernel4, requires_grad=False)]
        self.filter_kernels[0].requires_grad = False
        for i in range(self.num_kernels):
            self.pad_kernels.append(nn.ZeroPad2d((0, int(self.options['image_size'][0] - 3), 0, int(self.options['image_size'][1] - 3))))
        
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)
        m = m.unsqueeze(1)

        out_u =  []
        out_f = []
        for idx, filt in enumerate(self.filter_kernels):
            # convolution of input image u
            padding = 1
            u_idx = torch.zeros_like(u)
            for i in range(2):
                u_idx[..., i] = F.conv2d(u[..., i], filt.detach().unsqueeze(0).unsqueeze(0), padding=padding)
            out_u.append(u_idx)

            # create modified mri data
            pad = self.pad_kernels[idx]
            fft_kernel = torch.fft.fft2(pad(filt))
            f_complex = fft_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(f.size()[0], f.size()[1],f.size()[2], 1, 1) * torch.complex(f[..., 0], f[..., 1])
            f_idx = torch.zeros_like(f)
            f_idx[..., 0] = torch.real(f_complex)
            f_idx[..., 1] = torch.imag(f_complex)
            
            out_f.append(f_idx)
        
        out_u = torch.cat(out_u, dim=1)
        out_f = torch.cat(out_f, dim=1)
        output = {'u_t': out_u, 'f': out_f, 'coil_sens': c, 'sampling_mask': m, 'kernel': self.filter_kernels}
        return output
    
class FixedKernels(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.num_kernels = self.options['num_features']
        torch.autograd.set_detect_anomaly(True)

        self.pad_kernels = []
        filter_kernel1 =  torch.tensor(np.array([[0,0,0],[0,1,0],[0,0,0]])).to(options['device']).to(torch.float32)
        filter_kernel2 =  torch.tensor(np.array([[0,0,0],[0,-1,1],[0,0,0]])).to(options['device']).to(torch.float32)
        filter_kernel3 =  torch.tensor(np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])).to(options['device']).to(torch.float32)
        filter_kernel4 =  torch.tensor(np.array([[0,1,0],[1,-4,1],[0,1,0]])).to(options['device']).to(torch.float32)
        self.filter_kernels = [nn.Parameter(filter_kernel1, requires_grad=False), nn.Parameter(filter_kernel2, requires_grad=False), nn.Parameter(filter_kernel3, requires_grad=False), nn.Parameter(filter_kernel4, requires_grad=False)]
        self.filter_kernels[0].requires_grad = False
        for i in range(self.num_kernels):
            self.pad_kernels.append(nn.ZeroPad2d((0, int(self.options['image_size'][0] - 3), 0, int(self.options['image_size'][1] - 3))))
    
    def forward(self, inputs):
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)
        m = m.unsqueeze(1)

        out_u =  []
        out_f = []
        for idx, filt in enumerate(self.filter_kernels):
            # convolution of input image u
            padding = 1
            u_idx = torch.zeros_like(u)
            for i in range(2):
                u_idx[..., i] = F.conv2d(u[..., i], filt.detach().unsqueeze(0).unsqueeze(0), padding=padding)
            out_u.append(u_idx)

            # create modified mri data
            pad = self.pad_kernels[idx]
            fft_kernel = torch.fft.fft2(pad(filt))
            f_complex = fft_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(f.size()[0], f.size()[1],f.size()[2], 1, 1) * torch.complex(f[..., 0], f[..., 1])
            f_idx = torch.zeros_like(f)
            f_idx[..., 0] = torch.real(f_complex)
            f_idx[..., 1] = torch.imag(f_complex)
            
            out_f.append(f_idx)
        
        out_u = torch.cat(out_u, dim=1)
        out_f = torch.cat(out_f, dim=1)
        output = {'u_t': out_u, 'f': out_f, 'coil_sens': c, 'sampling_mask': m, 'kernel': self.filter_kernels}
        return output
   
class SharpenFilter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.num_kernels = self.options['num_features']
        self.filter_kernels = []
        self.pad_kernels = []
        filter_kernel =  torch.tensor(np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])).to(options['device']).to(torch.float32)
        self.filter_kernel = nn.Parameter(filter_kernel)
        self.filter_kernels.append(self.filter_kernel)
        self.pad_kernels.append(nn.ZeroPad2d((0, int(self.options['image_size'][0] - 3), 0, int(self.options['image_size'][1] - 3))))
        self.mask_sharpen_grad = torch.zeros_like(self.filter_kernel)
        self.mask_sharpen_grad[1, 1] = 1
        
    def forward(self, inputs):
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)
        m = m.unsqueeze(1)

        out_u =  []
        out_f = []
        if self.filter_kernel.grad is not None:
                self.filter_kernel.grad *= self.mask_sharpen_grad
                
        for idx, filt in enumerate(self.filter_kernels):
            # convolution of input image u
            print(filt)
            padding = idx + 1
            u_idx = torch.zeros_like(u)
            for i in range(2):
                u_idx[..., i] = F.conv2d(u[..., i], filt.detach().unsqueeze(0).unsqueeze(0), padding=padding)
            out_u.append(u_idx)

            # create modified mri data
            pad = self.pad_kernels[idx]
            fft_kernel = torch.fft.fft2(pad(filt))
            f_complex = fft_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(f.size()[0], f.size()[1],f.size()[2], 1, 1) * torch.complex(f[..., 0], f[..., 1])
            f_idx = torch.zeros_like(f)
            f_idx[..., 0] = torch.real(f_complex)
            f_idx[..., 1] = torch.imag(f_complex)
            out_f.append(f_idx)
            
        out_u = torch.cat(out_u, dim=1)
        out_f = torch.cat(out_f, dim=1)
        output = {'u_t': out_u, 'f': out_f, 'coil_sens': c, 'sampling_mask': m, 'kernel': self.filter_kernels}
        return output
    
class ApplyDerivative(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        self.num_kernels = self.options['num_features']
        self.derivatives = []
        self.image_size = self.options['image_size']
        
        self.xi = create_xi(self.image_size, options['net_type'])
        
    def forward(self, inputs):
        u = inputs['u_t']
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']
        u = u.unsqueeze(1)  # second entry will be the different feature images Nx1xHxWx2
        f = f.unsqueeze(1)  # Nx1xCxHxWx2
        c = c.unsqueeze(1)  # Nx1xCxHxW
        m = m.unsqueeze(1)  # Nx1xHxW

        out_u = [u]
        out_f = [f]        
        
        u_dx = ifftc2d(self.xi.unsqueeze(0).to(inputs['u_t'].device) * fftc2d(u, axis=[-3, -2]),axis=[-3, -2])
        out_u.append(u_dx)
        f_dx = self.xi.unsqueeze(0).unsqueeze(0).to(f.device) * f
        out_f.append(f_dx)
        out_u = torch.cat(out_u, dim=1)
        out_f = torch.cat(out_f, dim=1)
        output = {'u_t': out_u, 'f': out_f, 'coil_sens': c, 'sampling_mask': m, 'kernel': None}
        return output

class VariationalNetwork_learned_kernels(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        self.net_type = self.options['net_type']
        if self.net_type == 'original':
            self.options['num_features'] = 1
            cell_list = [OriginalRecon(**options)]
        if self.net_type == 'learned_kernels':
            self.options['num_features'] = 4
            cell_list = [FeatureExtraction(**options)]
        elif self.net_type =='1_kernel':
            self.options['num_features'] = 1
            cell_list = [FeatureExtraction_1kernel(**options)]
        elif self.net_type == 'random_kernels':
            self.options['num_features'] = 4
            cell_list = [RandomKernels(**options)]
        elif self.net_type == 'fixed_kernels':
            self.options['num_features'] = 4
            cell_list = [FixedKernels(**options)]
        elif self.net_type in ['original_dx_shared', 'original_dy_shared']:
            self.options['num_features'] = 2
            cell_list = [ApplyDerivative(**options)]
        elif self.net_type == 'sharpen_filter':
            cell_list = [SharpenFilter(**options)]
        for i in range(options['num_stages']):
            cell_list.append(VnMriReconCell(**options))

        self.cell_list = nn.Sequential(*cell_list)
        self.log_img_count = 0

    def forward(self, inputs):
        output = self.cell_list(inputs)
        return output['u_t'], output['kernel']

    def training_step(self, batch, batch_idx):
        recon_img, kernel = self(batch)
        if self.net_type in ['1_kernel','learned_kernels', 'random_kernels', 'fixed_kernels']:
            ref_img = self.get_reference(batch['reference'], kernel)
        # Todo: kernel train
        elif self.net_type in ['original_dx_shared', 'original_dy_shared']:
            ref_img = batch['reference'].unsqueeze(1)
            image_size = self.options['image_size']
            xi = create_xi(image_size, self.net_type)
            ref_img_dx = ifftc2d(xi.to(ref_img.device) * fftc2d(ref_img, axis=[-3, -2]),axis=[-3, -2])
            ref_img = torch.cat([ref_img, ref_img_dx], dim=1)

        if self.options['loss_type'] == 'complex':
            loss = F.mse_loss(recon_img, ref_img)
        elif self.options['loss_type'] == 'magnitude':
            recon_img_mag = torch_abs(recon_img)
            ref_img_mag = torch_abs(ref_img)
            loss = F.mse_loss(recon_img_mag,ref_img_mag) #/ (self.options['batch_size']*recon_img.shape[0])
        loss = self.options['loss_weight'] * loss
        return {'loss': loss}

    def test_step(self, batch, batch_idx, save=True, save_recon_np=False):
        recon_img, kernel = self(batch)
        if self.net_type in ['1_kernel','learned_kernels','random_kernels', 'fixed_kernels']:
            ref_img = torch.ones_like(recon_img)
            for i in range(self.options['num_features']):
                for j in range(2):
                    ref_img[:,i,:,:,j]  = torch.nn.functional.conv2d(batch['reference'][:,:,:,j].unsqueeze(1), kernel[i].unsqueeze(0).unsqueeze(0), padding=1).squeeze(1)

        elif self.net_type in ['original_dx_shared', 'original_dy_shared']:
            ref_img = batch['reference'].unsqueeze(1)
            image_size = self.options['image_size']
            xi = create_xi(image_size, self.net_type)
            ref_img_dx = ifftc2d(xi.to(ref_img.device) * fftc2d(ref_img,axis=[-3, -2]),axis=[-3, -2])
            ref_img = torch.cat([ref_img, ref_img_dx], dim=1)
        else:
            ref_img = batch['reference'].unsqueeze(1)
        if self.options['loss_type'] == 'complex':
            loss = F.mse_loss(recon_img, ref_img)
            ssim = []
            for i in range(recon_img.shape[0]):
                ssim.append(structural_similarity(ref_img[i,0,:,:,0].cpu().detach().numpy(), recon_img[i,0,:,:,0].cpu().detach().numpy()))
        elif self.options['loss_type'] == 'magnitude':    
            recon_img_mag = torch_abs(recon_img)
            ref_img_mag = torch_abs(ref_img)
            #loss = mse_varnet(recon_img_mag, ref_img_mag)
            loss = F.mse_loss(recon_img_mag,ref_img_mag) #/ (self.options['batch_size']*recon_img.shape[0])
            ssim = []
            for i in range(recon_img.shape[0]):
                ssim.append(structural_similarity(ref_img_mag[i,:,:].cpu().detach().numpy(), recon_img_mag[i,:,:].cpu().detach().numpy()))
        
        if save:
            for i in range(self.options['num_features']):
                img_save_dir = Path(self.options['save_dir']) / (f'eval_result_img_{self.net_type}_{i}_' + self.options['name'])
                img_save_dir.mkdir(parents=True, exist_ok=True)
                save_recon(batch['u_t'], recon_img[:,i,:,:,:], ref_img[:,i,:,:,:], batch_idx, img_save_dir, self.options['error_scale'], True, rescale=True)
                if save_recon_np:
                    img_save_dir = Path(self.options['save_dir']) / (f'eval_result_img_{self.net_type}_{i}_' + self.options['name']+'_recon')
                    img_save_dir.mkdir(parents=True, exist_ok=True)
                    for j in range(recon_img.shape[0]):
                        np_img = np.concatenate((recon_img[j,i,:,:,:].unsqueeze(0).detach().cpu().numpy(),ref_img[j,i,:,:,:].unsqueeze(0).detach().cpu().numpy()),axis=0)
                        np.save(str(img_save_dir) + f'/recon_{batch_idx+j}.npy', np_img)
        return {'test_loss': loss.item(), 'recon_img': recon_img, 'ssim':np.mean(ssim)}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def configure_optimizers(self):
        if self.options['optimizer'] == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.options['lr'])
        elif self.options['optimizer'] == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.options['lr'], momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.options['lr'], momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'iipg':
            iipg = IIPG(torch.optim.SGD, self.parameters(), lr=self.options['lr'], momentum=self.options['momentum'])
            return iipg
        
    def get_reference(self, references, kernel):
        if self.net_type in ['1_kernel','learned_kernels','random_kernels','fixed_kernels']:
            ref_img = torch.ones(references.shape[0],self.options['num_features'],self.options['image_size'][0],self.options['image_size'][1],2).to(references.device)
            for i in range(self.options['num_features']):
                for j in range(2):
                    ref_img[:,i,:,:,j]  = torch.nn.functional.conv2d(references[:,:,:,j].unsqueeze(1), kernel[i].unsqueeze(0).unsqueeze(0), padding=1).squeeze(1)
        elif self.net_type in ['original_dx_shared', 'original_dy_shared']:
            ref_img = references.unsqueeze(1)
            image_size = self.options['image_size']
            xi = create_xi(image_size, self.net_type)
            ref_img_dx = ifftc2d(xi.to(ref_img.device) * fftc2d(ref_img, axis=[-3, -2]),axis=[-3, -2])
            ref_img = torch.cat([ref_img, ref_img_dx], dim=1)
        else:
            ref_img = references.unsqueeze(1)
        return ref_img

def create_xi(image_size, net_type):
    xi_re_im = torch.zeros(1, image_size[0], image_size[1], 2)
    if net_type == 'original_dx_shared':
        xi = torch.exp(2 * 1j * np.pi * torch.linspace(1, image_size[0], image_size[0])) - 1
        xi = xi.repeat(image_size[1],1)
            
    elif net_type == 'original_dy_shared':
        xi = torch.exp(2 * 1j * np.pi * torch.linspace(1, image_size[1], image_size[1])) - 1
        xi = torch.transpose(xi.repeat(image_size[0],1),0,1)
    else: 
        raise ValueError('Unknown net_type. Use dx or dy.')
    xi_re_im[0,:,:,0] = xi.real
    xi_re_im[0,:,:,1] = xi.imag
    return xi_re_im
