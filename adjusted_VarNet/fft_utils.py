import torch
import numpy as np
import sys
import warnings

""" FFT, iFFT, and FFTshift function """
def generic_fftshift(x,axis=[-2,-1],inverse=False):
    """
    Fourier shift to center the low frequency components

    Parameters
    ----------
    x : torch Tensor
        Input array
    inverse : bool
        whether the shift is for fft or ifft

    Returns
    -------
    shifted array

    """
    if len(axis) > len(x.shape):
        raise ValueError('Not enough axis to shift around!')
    
    y = x
    for axe in axis:
        dim_size = x.shape[axe]
        shift = int(dim_size/2)
        if inverse:
            if not dim_size%2 == 0:
                shift += 1
        
        y = torch.roll(y,shift,axe)
    
    return y

def fftshift(x,axis=[-2,-1]):
    return generic_fftshift(x,axis=axis,inverse=False)

def ifftshift(x,axis=[-2,-1]):
    return generic_fftshift(x,axis=axis,inverse=True)

def fftc2d(x, axis=[-2,-3]):
    """
    Centered 2d Fourier transform, performed on axis(-2,-3)

    Parameters
    ----------
    x : input array
        array to perform fft on
        shape: [NxWxHx2]: real and imaginary channel

    Returns
    -------
    centered fft transformed of x

    """
    if (axis == [-2,-3] or axis==[-3,-2]) and x.shape[-1]==2:
        x = torch.complex(x[..., 0], x[..., 1])
        #x = ifftshift(x, axis=(-2,-1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        #x = ifftshift(x, axis=(-2,-1))
        x = torch.cat((x.real.unsqueeze(-1), x.imag.unsqueeze(-1)), dim=-1)
    elif (axis == [-3,-4] or axis==[-4,-3]) and x.shape[-2]==2:
        x = torch.complex(x[..., 0,:], x[..., 1,:])
        #x = ifftshift(x, axis=(-3,-2))
        x = torch.fft.fft2(x, dim=(-3, -2), norm="ortho")
        #x = fftshift(x, axis=(-3,-2))
        x = torch.cat((x.real.unsqueeze(-2), x.imag.unsqueeze(-2)), dim=-2)
    #x = ifftshift(x, axis=(-3,-2))
    #x = torch.fft.fft2(x,dim=(-3,-2),norm="ortho")
    #x = fftshift(x,axis=(-2,-3))
    return x

def ifftc2d(x,axis=[-2,-3]):
    """
    Centered inverse 2d Fourier transform, performed on axis(-2,-3)

    Parameters
    ----------
    x : input array
        array to perform ifft on
        shape: [NxWxHx2]: real and imaginary channel

    Returns
    -------
    centered ifft transformed of x

    """ 
    if (axis == [-2,-3] or axis==[-3,-2]) and x.shape[-1]==2:
        x = torch.complex(x[..., 0], x[..., 1])
        #x = ifftshift(x, axis=(-2,-1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
        #x = fftshift(x, axis=(-2,-1))
        x = torch.cat((x.real.unsqueeze(-1), x.imag.unsqueeze(-1)), dim=-1)
    elif (axis == [-3,-4] or axis==[-4,-3]) and x.shape[-2]==2:
        x = torch.complex(x[..., 0,:], x[..., 1,:])
        #x = ifftshift(x, axis=(-3,-2))
        x = torch.fft.ifft2(x, dim=(-3, -2), norm="ortho")
        #x = fftshift(x, axis=(-3,-2))
        x = torch.cat((x.real.unsqueeze(-2), x.imag.unsqueeze(-2)), dim=-2)
    #x = ifftshift(x,axis=[-2,-3])
    #x = torch.fft.ifft2(x,dim=(-3,-2),norm="ortho")
    #x = fftshift(x,axis=[-2,-3])
    return x

def torch_abs(x, axis=-1):
    """
    Compute magnitude for two-channel complex torch tensor
    """
    mag = torch.sqrt(torch.sum(torch.square(x),axis=axis,keepdim=False) + 1e-9)
    return mag

def mse_varnet(x, y):
    loss = torch.sum(torch.square(x-y))
    return loss / (2 * x.shape[0])

""" Converting to and from complex image and two channels image """
def real_2_complex(x):
    """
    Convert real-valued, 1-channel, torch tensor to complex-valued, 2-channel
    with 0 imaginary component

    Parameters
    ----------
    x : input tensor

    Returns
    -------
    complex array with 2-channel at the end

    """
    out = x.squeeze()
    out = x.unsqueeze(-1)
    imag = torch.zeros(out.shape,dtype=out.dtype,requires_grad=out.requires_grad)
    out = torch.cat((out,imag),dim=-1)
    return out

def complex_2_numpy(x):
    """
    Convert 2-channel complex torch tensor to numpy complex number

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    out = x.numpy()
    out = np.take(out,0,axis=-1) + np.take(out,1,axis=-1)*1j
    return out

def numpy_2_complex(x):
    """
    Convert numpy complex array to 2-channel complex torch tensor

    Parameters
    ----------
    x : numpy complex array
        input array

    Returns
    -------
    Equivalent 2-channel torch tensor

    """
    real = np.real(x)
    real = np.expand_dims(real,-1)
    imag = np.imag(x)
    imag = np.expand_dims(imag,-1)
    out = np.concatenate((real,imag),axis=-1)
    out = torch.from_numpy(out)
    return out

def conj(x, dim=-1):
    """
    Calculate the complex conjugate of x
    
    x is two-channels complex torch tensor
    """
    if dim==-1:
        assert x.shape[dim] == 2
        return torch.stack((x[..., 0], -x[..., 1]), dim=-1)
    elif dim==-2:
        assert x.shape[dim] == 2
        return torch.stack((x[..., 0, :], -x[..., 1, :]), dim=-2)
    
def mult_i(x):
    """
    Multiply torch tensor x with i
    x is two-channels complex torch tensor
    """
    assert x.shape[-1] == 2
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1)


def complex_mul(x,y):
    """ Complex multiply 2-channel complex torch tensor x,y
    """
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)

def complex_mul_feat(x,y):
    """ Complex multiply 2-channel complex torch tensor x,y for features
    """
    assert x.shape[-2] == y.shape[-2] == 2
    re = x[..., 0,:] * y[..., 0,:] - x[..., 1,:] * y[..., 1,:]
    im = x[..., 0,:] * y[..., 1,:] + x[..., 1,:] * y[..., 0,:]
    return torch.stack((re, im), dim=-2)
