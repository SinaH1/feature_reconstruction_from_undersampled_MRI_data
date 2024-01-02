import numpy as np
import matplotlib.pyplot as plt

def mse(x,y):
    return np.square(x - y).mean()

def derivative_fourier(xi, recon, axis=(-2, -1)):
    return np.fft.ifft2(xi * np.fft.fft2(recon, axes=axis,norm='ortho'), axes=axis ,norm='ortho')
    #return np.fft.ifft2(np.fft.ifftshift(xi * np.fft.fftshift(np.fft.fft2(recon, axes=axis),axes=axis), axes=axis),axes=axis)
    #return np.fft.ifft2(np.fft.fftshift(xi, axes=axis) * np.fft.fft2(recon, axes=axis,norm='ortho'), axes=axis, norm='ortho')

def derivative_finite_differences(recon, axis=0):
    if axis==1:
        boundary = np.expand_dims(recon[:,-1], axis=1)
    else:
        boundary = np.expand_dims(recon[-1,:], axis=0)
    return np.diff(recon, axis=axis, append=boundary)

def derivative_finite_differences_central(recon, axis=1):
    if axis==1:
        recon_d = np.concatenate((recon[:,1:2,:]-recon[:,-1:,:], recon[:,2:,:] - recon[:,:-2,:], recon[:,0:1,:] - recon[:,-2:-1,:]),axis=1)/2
    elif axis==2:
        recon_d = np.concatenate((recon[:, :, 1:2] - recon[:, :, -1:], recon[:,:, 2:] - recon[:, :, :-2],recon[:, :, 0:1] - recon[:, :, -2:-1]), axis=2) / 2
    return recon_d

def operator_A(img, coilsens, mask):
    """ Forward MRI Cartesian Operator """
    return np.fft.fft2(coilsens * img, norm='ortho')*mask

def adjoint_operator_A(rawdata, coilsens, mask):
    """ Adjoint MRI Cartesian Operator """
    return np.sum(np.fft.ifft2(rawdata * mask, norm='ortho')*np.conj(coilsens), axis=0)

def squared_norm(x):
    return np.sum(x.real**2 + x.imag**2)

def normalize(x):
    x= np.abs(x)
    if np.max(x) != 0:
        return (x/np.max(x)).astype(np.float64)
    else:
        return x.astype(np.float64)

def CGNE(f, coils, mask, max_iter=10, discrepancy_threshold=1e-1):
    x = np.zeros(np.shape(f)[1:]).astype(complex)
    d = f-operator_A(x, coils, mask)
    p = adjoint_operator_A(d, coils, mask)
    s_old = p
    discrepancy_threshold = discrepancy_threshold * np.sqrt(squared_norm(f))
    for i in range(max_iter):
        q = operator_A(p, coils, mask)
        alpha = (squared_norm(s_old) / squared_norm(q))
        x += alpha*p
        d-= alpha*q
        s_new = adjoint_operator_A(d, coils, mask)
        beta = squared_norm(s_new) / squared_norm(s_old)
        p = s_new + beta * p
        s_old = s_new
        discrepancy = np.sqrt(squared_norm(f - operator_A(x, coils, mask)))
        print(discrepancy)
        if discrepancy <= discrepancy_threshold:
            print(f"Discrepancy principle at iteration {i + 1}.")
        #    break
    return x

def CGNE_exact_old(f, f_dx, coils, mask, coil_dx, max_iter=10, axes=[-2,-1], discrepancy_threshold=1e-1):
    x = np.zeros(np.shape(f)[1:]).astype(complex)
    x_dx = np.zeros(np.shape(f)[1:]).astype(complex)
    d = f - operator_A(x, coils, mask)
    d_dx = f_dx - mask * np.fft.fft2(coil_dx * adjoint_operator_A(f, coils, mask), axes=axes,norm='ortho') - operator_A(x_dx, coils, mask)
    p = adjoint_operator_A(d, coils, mask)
    p_dx = adjoint_operator_A(d_dx, coils, mask)
    s_old = p
    discrepancy_threshold = discrepancy_threshold * np.sqrt(squared_norm(f_dx - operator_A(x_dx, coils, mask), axes=axes,norm='ortho'))
    #print(discrepancy_threshold)
    for i in range(max_iter):
        q = operator_A(p, coils, mask)
        q_dx = operator_A(p_dx, coils, mask)
        alpha = (squared_norm(s_old) / squared_norm(q))
        alpha_dx = (squared_norm(s_old) / squared_norm(q_dx))
        x += alpha * p
        x_dx += alpha_dx * p_dx
        d -= alpha * q
        d_dx -= alpha*q_dx
        s_new = adjoint_operator_A(d, coils, mask)
        beta = squared_norm(s_new) / squared_norm(s_old)
        p = s_new + beta * p
        s_old = s_new
        discrepancy = np.sqrt(squared_norm(f - operator_A(x, coils, mask)))
        # print(discrepancy)
        if discrepancy <= discrepancy_threshold:
            print(f"Discrepancy principle at iteration {i + 1}.")
            break
    return x

def CGNE_exact(f, coils, mask, coil_dx, max_iter=10, axes=[-2,-1], discrepancy_threshold=1e-1):
    x = np.zeros(np.shape(f)[1:]).astype(complex)
    d = f - mask * np.fft.fft2(coil_dx * adjoint_operator_A(f, coils, mask), axes=axes,norm='ortho')-operator_A(x, coils, mask)
    p = adjoint_operator_A(d, coils, mask)
    s_old = p
    discrepancy_threshold = discrepancy_threshold * np.sqrt(squared_norm(f_dx - mask * np.fft.fft2(coil_dx * adjoint_operator_A(f, coils, mask), axes=axes,norm='ortho')))
    #print(discrepancy_threshold)
    for i in range(max_iter):
        q = operator_A(p, coils, mask)
        alpha = (squared_norm(s_old) / squared_norm(q))
        x += alpha*p
        d-= alpha*(q - mask * np.fft.fft2(coil_dx * p, axes=axes,norm='ortho'))
        s_new = adjoint_operator_A(d, coils, mask)
        beta = squared_norm(s_new) / squared_norm(s_old)
        p = s_new + beta * p
        s_old = s_new
        discrepancy = np.sqrt(squared_norm(f - operator_A(x, coils, mask)))
        # print(discrepancy)
        if discrepancy <= discrepancy_threshold:
            print(f"Discrepancy principle at iteration {i + 1}.")
            break
    return x

def CGNE_plot(f, coils, mask, max_iter=10):
    x = [np.zeros_like(mask).astype(complex)]
    d = f-operator_A(x[-1], coils, mask)
    p = adjoint_operator_A(d, coils, mask)
    s_old = p

    for i in range(max_iter):
        q = operator_A(p, coils, mask)
        alpha = (squared_norm(s_old) / squared_norm(q))
        x.append(x[-1] + alpha*p)
        d-= alpha*q
        s_new = adjoint_operator_A(d, coils, mask)
        beta = squared_norm(s_new) / squared_norm(s_old)
        p = s_new + beta * p
        s_old = s_new
    return x


def mriAdjointOp(rawdata, coilsens, mask):
    """ Adjoint MRI Cartesian Operator """
    return np.sum(np.fft.ifft2(rawdata * mask)*np.conj(coilsens), axis=0)

def mriForwardOp(img, coilsens, mask):
    """ Forward MRI Cartesian Operator """
    return fft2c(coilsens * img)*mask

def landweber_plot(f, coils, mask, max_iter=100, axes=[-2,-1], discrepancy_threshold=1e-1):
    u = [np.sum(np.fft.ifft2(f * mask, axes=axes) * np.conj(coils), axis=0)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coils * u[-1], norm='ortho', axes=axes) * mask
        u.append(u[-1] - 1 * np.sum(np.fft.ifft2((u1 - f) * mask, norm='ortho', axes=axes) * np.conj(coils), axis=0))
    return u

def landweber_exact_plot(f, f_dx, coils, mask, coil_dx, max_iter=100, axes=[-2,-1], discrepancy_threshold=1e-1):
    u = np.sum(np.fft.ifft2(f * mask, axes=axes) * np.conj(coils), axis=0)
    u_dx = [np.sum(np.fft.ifft2(f_dx * mask, axes=axes) * np.conj(coils), axis=0)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coils * u, norm='ortho', axes=axes) * mask
        u -= 1 * np.sum(np.fft.ifft2((u1 - f) * mask, norm='ortho', axes=axes) * np.conj(coils), axis=0)
        u1_dx = np.fft.fft2(coils * u_dx[-1], norm='ortho', axes=axes) * mask
        u_dx.append(u_dx[-1] -  1 * np.sum(np.fft.ifft2((u1 - (f_dx - mask * np.fft.fft2(coil_dx * u, axes=axes, norm='ortho')) * mask), norm='ortho', axes=axes) * np.conj(coils), axis=0))
    return u_dx

def landweber(f, coils, mask, max_iter=100, axes=[-2,-1], discrepancy_threshold=np.infty):
    u = np.zeros((f.shape[1], f.shape[2])).astype(np.complex64)
    #discrepancy_threshold = discrepancy_threshold * np.sqrt(squared_norm(f))
    for i in range(max_iter):
        u1 = np.fft.fft2(coils * u, norm='ortho', axes=axes) * mask
        u -= 1 * np.sum(np.fft.ifft2((u1 - f) * mask, norm='ortho', axes=axes) * np.conj(coils), axis=0)
        discrepancy = np.sqrt(squared_norm(f - operator_A(u, coils, mask)))
        print(discrepancy)
        if discrepancy <= discrepancy_threshold:
            print(f"Discrepancy principle at iteration {i + 1}.")
            break
    return u, i

def landweber_exact(f, f_dx, coils, mask, coil_dx, max_iter=100, axes=[-2,-1], discrepancy_threshold=1e-1):
    #u = np.sum(np.fft.ifft2(f * mask, axes=axes) * np.conj(coils), axis=0)
    u = np.zeros((f.shape[1], f.shape[2])).astype(np.complex64)
    #u_dx = np.sum(np.fft.ifft2(f_dx * mask, axes=axes) * np.conj(coils), axis=0)
    u_dx = np.zeros((f.shape[1], f.shape[2])).astype(np.complex64)
    discrepancy_threshold = discrepancy_threshold * np.sqrt(squared_norm(f_dx - mask * np.fft.fft2(coil_dx * adjoint_operator_A(f, coils, mask), axes=axes, norm='ortho')))

    for i in range(max_iter):
        u1 = np.fft.fft2(coils * u, norm='ortho', axes=axes) * mask
        u -= 1 * np.sum(np.fft.ifft2((u1 - f) * mask, norm='ortho', axes=axes) * np.conj(coils), axis=0)
        u1_dx = np.fft.fft2(coils * u_dx, norm='ortho', axes=axes) * mask
        u_dx -= 1 * np.sum(np.fft.ifft2((u1 - (f_dx - mask * np.fft.fft2(coil_dx * u, axes=axes, norm='ortho')) * mask), norm='ortho', axes=axes) * np.conj(coils), axis=0)
        discrepancy = np.sqrt(squared_norm(f_dx - mask * np.fft.fft2(coil_dx * u, axes=axes, norm='ortho') - operator_A(u_dx, coils, mask)))
        # print(discrepancy)
        if discrepancy <= discrepancy_threshold:
            print(f"Discrepancy principle at iteration {i + 1}.")
            break
    return u_dx

def show_img(u_show,title=''):
    if u_show.dtype == np.complex64 or u_show.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        u_show = np.abs(u_show)
    plt.figure()
    plt.imshow(u_show, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_kspace(f,title=''):
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(f[0])), cmap='gray')
    plt.title(title)
    plt.show()

'''def get_mse_psnr_ssim(recon, img, type='dx'):
    xi = np.linspace(-int(img.shape[0] / 2), int((img.shape[0] - 1) / 2), img.shape[0])
    # xi = np.linspace(-96, 95, img.shape[0])*np.pi/192
    if type == 'dx':
        axis_fd=1
        xi = np.repeat(np.expand_dims(xi, axis=0), img.shape[0], axis=0)

    if type == 'dy':
        axis_fd=0
        xi = np.repeat(np.expand_dims(xi, axis=1), img.shape[1], axis=1)
    
    if type in ['dx','dy']:
        recon_fourier = derivative_fourier(xi, recon)
        recon_fd = derivative_finite_differences(recon, axis=axis_fd)

        recon_mse_fourier = mse(np.abs(recon), np.abs(img))
    recon_mse_fd = mse(np.abs(recon_dx) + np.abs(recon_dy), np.abs(img_fd_dx) + np.abs(img_fd_dy))
    recon_fourier_mse = mse(np.abs(recon_fourier_dx) + np.abs(recon_fourier_dy),
                               np.abs(img_fourier_dx) + np.abs(img_fourier_dy))
    recon_fd_mse = mse(np.abs(recon_fd_dx) + np.abs(recon_fd_dy), np.abs(img_fd_dx) + np.abs(img_fd_dy))

    recon = (np.abs(recon_dx) + np.abs(recon_dy)) / np.max(np.abs(recon_dx) + np.abs(recon_dy))
    img_fourier = (np.abs(img_fourier_dx) + np.abs(img_fourier_dy)) / np.max(
        np.abs(img_fourier_dx) + np.abs(img_fourier_dy))
    img_fd = (np.abs(img_fd_dx) + np.abs(img_fd_dy)) / np.max(np.abs(img_fd_dx) + np.abs(img_fd_dy))
    recon_deriv_fourier = (np.abs(recon_fourier_dx) + np.abs(recon_fourier_dy)) / np.max(
        np.abs(recon_fourier_dx) + np.abs(recon_fourier_dy))
    recon_deriv_fd = (np.abs(recon_fd_dx) + np.abs(recon_fd_dy)) / np.max(np.abs(recon_fd_dx) + np.abs(recon_fd_dy))

    recon_psnr_fourier = peak_signal_noise_ratio(img_fourier, recon)
    recon_psnr_fd = peak_signal_noise_ratio(img_fd, recon)
    recon_fourier_psnr = peak_signal_noise_ratio(img_fourier, recon_deriv_fourier)
    recon_fd_psnr = peak_signal_noise_ratio(img_fd, recon_deriv_fd)

    recon_ssim_fourier = structural_similarity(img_fourier, recon)
    recon_ssim_fd = structural_similarity(img_fd, recon)
    recon_fourier_ssim = structural_similarity(img_fourier, recon_deriv_fourier)
    recon_fd_ssim = structural_similarity(img_fd, recon_deriv_fd)'''
