import sys
sys.path.insert(0,'../')
import argparse
from utils import *
from skimage.metrics import structural_similarity
from scipy import signal

parser = argparse.ArgumentParser(description='Correction term arguments')
parser.add_argument('--operator', type=str, default='TF', help='Operator for partial derivative of image. Choices are: TF (Truncated Fourier series), FD (Forward Differences), Convolution')
parser.add_argument('--task', type=str, default='r_coil_approximation_residual', help='Task from . Choices are: r_coil_approximation_residual, e_correction_term, d_landweber, d_CG')

def main():
    args = parser.parse_args()
    args = vars(args)
    operator = args['operator']  # FD, Convolution, Fourier
    if args['task'] == 'r_coil_approximation_residual':
        correction_term_approximation(operator=operator)
    elif args['task'] == 'e_correction_term':
        correction_term(operator=operator)
    elif args['task'] == 'd_landweber':
        correction_term_landweber(operator=operator)
    elif args['task'] == 'd_CG':
        correction_term_CG(operator=operator)

def correction_term_approximation(operator='TF', max_iter=10, step_size=1):
    img = np.load('../dataset/images/image_1007.npy').squeeze(2)
    mask = np.load('../dataset/sampling_pattern.npy').squeeze(2)
    coil = np.transpose(np.load('../dataset/coil_sensitivities/coil_sensitivities_1007.npy'), axes=[2, 0, 1]).astype(np.complex64)
    img /= np.max(img)
    g = np.fft.fft2(coil * img[np.newaxis, :, :], norm='ortho', axes=(1, 2))
    g *= mask
    coil_dx = np.diff(coil, axis=1, append=np.expand_dims(coil[:, -1, :], axis=1))
    coil_dy = np.diff(coil, axis=2, append=np.expand_dims(coil[:, :, -1], axis=2))

    # landweber method
    u_recon_landweber = [np.zeros((g.shape[1], g.shape[2])).astype(np.complex64)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coil * u_recon_landweber[-1], norm='ortho', axes=(-2, -1)) * mask
        u_recon_landweber.append(u_recon_landweber[-1] - step_size * np.sum(
            np.fft.ifft2((u1 - g) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))

    # CG method
    u_recon_cg = CGNE_plot(g, coil, mask, max_iter=max_iter)

    # correction terms
    if operator == 'TF':
        c_landweber_dx = mask * np.fft.fft2(coil_dx * u_recon_landweber[-1][np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_cg_dx = mask * np.fft.fft2(coil_dx * u_recon_cg[-1][np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_true_dx = mask * np.fft.fft2(coil_dx * img[np.newaxis,:,:], norm='ortho', axes=(-2, -1))

        c_landweber_dy = mask * np.fft.fft2(coil_dy * u_recon_landweber[-1][np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_cg_dy = mask * np.fft.fft2(coil_dy * u_recon_cg[-1][np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_true_dy = mask * np.fft.fft2(coil_dy * img[np.newaxis,:,:], norm='ortho', axes=(-2, -1))

    elif operator in ['FD', 'Convolution']:
        c_landweber_dx = mask * np.fft.fft2(coil_dx * np.roll(u_recon_landweber[-1],-1, axis=0)[np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_cg_dx = mask * np.fft.fft2(coil_dx * np.roll(u_recon_cg[-1],-1, axis=0)[np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_true_dx = mask * np.fft.fft2(coil_dx * np.roll(img,-1, axis=0)[np.newaxis,:,:], norm='ortho', axes=(-2, -1))

        c_landweber_dy = mask * np.fft.fft2(coil_dy * np.roll(u_recon_landweber[-1],-1, axis=1)[np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_cg_dy = mask * np.fft.fft2(coil_dy * np.roll(u_recon_cg[-1],-1, axis=1)[np.newaxis,:,:], norm='ortho', axes=(-2, -1))
        c_true_dy = mask * np.fft.fft2(coil_dy * np.roll(img,-1, axis=1)[np.newaxis,:,:], norm='ortho', axes=(-2, -1))

    # residuum
    res_landweber_dx = np.sqrt(np.sum(np.abs(c_landweber_dx - c_true_dx) ** 2))
    res_cg_dx = np.sqrt(np.sum(np.abs(c_cg_dx - c_true_dx) ** 2))
    norm_dx = np.sqrt(np.sum(np.abs(c_true_dx) ** 2))

    res_landweber_dy = np.sqrt(np.sum(np.abs(c_landweber_dy - c_true_dy) ** 2))
    res_cg_dy = np.sqrt(np.sum(np.abs(c_cg_dy - c_true_dy) ** 2))
    norm_dy = np.sqrt(np.sum(np.abs(c_true_dy) ** 2))

    print(f'dx with {operator}')
    #print('||c_landweber-c_true||   ', res_landweber_dx)
    #print('||c_cg-c_true||          ', res_cg_dx)
    #print('||c_true||               ', norm_dx, '\n')
    #print('||c_landweber-c_true||/||c_true||', res_landweber_dx / norm_dx)
    print('||c_cg-c_true||/||c_true||', res_cg_dx / norm_dx, '\n')
    print(f'dy with {operator}')
    #print('||c_landweber-c_true||   ', res_landweber_dy)
    #print('||c_cg-c_true||          ', res_cg_dy)
    #print('||c_true||               ', norm_dy, '\n')
    #print('||c_landweber-c_true||/||c_true||', res_landweber_dy / norm_dy)
    print('||c_cg-c_true||/||c_true||', res_cg_dy / norm_dy, '\n')


def correction_term(operator = 'TF'): ## Verwendung coil_dx für Fourier nach FD, Randbedingung für FD?
    img = np.load('../dataset/images/image_1007.npy').squeeze(2)
    mask = np.load('../dataset/sampling_pattern.npy').squeeze(2)
    coil = np.transpose(np.load('../dataset/coil_sensitivities/coil_sensitivities_1007.npy'), axes=[2, 0, 1]).astype(np.complex64)
    img /= np.max(img)
    g = np.fft.fft2(coil * img[np.newaxis, :, :], norm='ortho', axes=(1, 2))

    g *= mask
    if operator == 'FD':
        xi = np.exp(2 * 1j * np.pi * np.linspace(0, 191, 192) / 192) - 1
        xi_dx = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
        xi_dy = xi[np.newaxis, np.newaxis, :].astype(np.complex64)

    elif operator == 'TF':
        xi = 2 * np.pi * 1j * np.linspace(-96, 95, 192) / 192
        xi= np.fft.fftshift(xi)
        xi_dx = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
        xi_dy = xi[np.newaxis, np.newaxis, :].astype(np.complex64)

    elif operator =='Convolution':
        dx_kernel = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        dy_kernel = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        padded_dx_kernel = np.pad(-dx_kernel, (0, 189), mode='constant')
        padded_dy_kernel = np.pad(-dy_kernel, (0, 189), mode='constant')
        xi_dx = 192 * np.fft.fft2(padded_dx_kernel, norm='ortho')
        xi_dy = 192 * np.fft.fft2(padded_dy_kernel, norm='ortho')

    g_dx = xi_dx * g
    g_dy = xi_dy * g

    coil_dx = np.diff(coil, axis=1, append=np.expand_dims(coil[:, -1, :], axis=1))
    coil_dy = np.diff(coil, axis=2, append=np.expand_dims(coil[:, :, -1], axis=2))
    if operator == 'TF':
        correction_dx = mask * np.fft.fft2(coil_dx * img, axes=(1, 2), norm='ortho')
        correction_dy = mask * np.fft.fft2(coil_dy * img, axes=(1, 2), norm='ortho')

    elif operator in ['FD','Convolution']:
        correction_dx = mask * np.fft.fft2(coil_dx *  np.roll(img, -1, axis=0), axes=(1, 2), norm='ortho')
        correction_dy = mask * np.fft.fft2(coil_dy * np.roll(img, -1, axis=1), axes=(1, 2), norm='ortho')

    E_dx = mask * np.fft.fft2(correction_dx, norm='ortho')
    E_dy = mask * np.fft.fft2(correction_dy, norm='ortho')

    dx = np.sqrt(np.sum(np.abs(E_dx) ** 2)) / np.sqrt(np.sum(np.abs(g_dx) ** 2))
    dy = np.sqrt(np.sum(np.abs(E_dy) ** 2)) / np.sqrt(np.sum(np.abs(g_dy) ** 2))

    print(f'dx with {operator}')
    print('||c_true||/||Dv||', dx, '\n')
    print(f'dy with {operator}')
    print('||c_true||/||Dv||', dy)

def correction_term_CG(operator = 'TF', max_iter = 10):
    img = np.load('../dataset/images/image_1007.npy').squeeze(2)
    mask = np.load('../dataset/sampling_pattern.npy').squeeze(2)
    coil = np.transpose(np.load('../dataset/coil_sensitivities/coil_sensitivities_1007.npy'), axes=[2, 0, 1]).astype(np.complex64)
    img /= np.max(img)
    g = np.fft.fft2(coil * img[np.newaxis, :, :], norm='ortho', axes=(1, 2))

    g *= mask
    if operator == 'FD':
        xi = np.exp(2 * 1j * np.pi * np.linspace(0, 191, 192) / 192) - 1
        xi_dx = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
        xi_dy = xi[np.newaxis, np.newaxis, :].astype(np.complex64)
    elif operator == 'TF':
        xi = 2 * np.pi * 1j * np.linspace(-96, 95, 192) / 192
        xi = np.fft.fftshift(xi)
        xi_dx = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
        xi_dy = xi[np.newaxis, np.newaxis, :].astype(np.complex64)
    elif operator =='Convolution':
        dx_kernel = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        dy_kernel = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        padded_dx_kernel = np.pad(dx_kernel, (0, 189), mode='constant')
        padded_dy_kernel = np.pad(dy_kernel, (0, 189), mode='constant')
        xi_dx = 192 * np.fft.fft2(padded_dx_kernel, norm='ortho')
        xi_dy = 192 * np.fft.fft2(padded_dy_kernel, norm='ortho')

    coil_dx = np.diff(coil, append=np.expand_dims(coil[:, -1, :], axis=1), axis=1)
    coil_dy = np.diff(coil, append=np.expand_dims(coil[:, :, -1], axis=2), axis=2)

    recon_CG = CGNE_plot(g, coil, mask, max_iter=max_iter)
    g_dx = xi_dx * g
    g_dy = xi_dy * g
    if operator == 'TF':
        corr_dx = mask * np.fft.fft2(coil_dx * recon_CG[-1][np.newaxis,:,:], norm='ortho', axes=(1,2))
        corr_dy = mask * np.fft.fft2(coil_dy * recon_CG[-1][np.newaxis,:,:], norm='ortho', axes=(1,2))

    elif operator in ['FD','Convolution']:
        corr_dx = mask * np.fft.fft2(coil_dx * np.roll(recon_CG[-1],-1, axis=0)[np.newaxis,:,:], norm='ortho', axes=(1,2))
        corr_dy = mask * np.fft.fft2(coil_dy * np.roll(recon_CG[-1],-1, axis=1)[np.newaxis,:,:], norm='ortho', axes=(1,2))

    dxrecon_CG_without = CGNE_plot(g_dx, coil, mask, max_iter=max_iter)
    dxrecon_CG_with = CGNE_plot(g_dx - corr_dx, coil, mask, max_iter=max_iter)
    dyrecon_CG_without = CGNE_plot(g_dy, coil, mask, max_iter=max_iter)
    dyrecon_CG_with = CGNE_plot(g_dy - corr_dy, coil, mask, max_iter=max_iter)

    norm_x = np.sqrt(np.sum(np.abs(dxrecon_CG_without[-1] - dxrecon_CG_with[-1]) ** 2)) / np.sqrt(np.sum(np.abs(dxrecon_CG_with[-1]) ** 2))
    norm_y = np.sqrt(np.sum(np.abs(dyrecon_CG_without[-1] - dyrecon_CG_with[-1]) ** 2)) / np.sqrt(np.sum(np.abs(dyrecon_CG_with[-1]) ** 2))

    print(f'dx with {operator}')
    print('||Lu_incl - Lu_omit||/||Lu_incl||', norm_x, '\n')
    print(f'dy with {operator}')
    print('||Lu_incl - Lu_omit||/||Lu_incl||', norm_y, '\n')


    if operator =='FD':
        recon_CG_dx = np.diff(recon_CG[-1], axis=0, append=np.expand_dims(recon_CG[-1][-1, :], axis=0))
        recon_CG_dy = np.diff(recon_CG[-1], axis=1, append=np.expand_dims(recon_CG[-1][:, -1], axis=1))

        img_dx = np.diff(img, axis=0, append=np.expand_dims(img[-1, :], axis=0))
        img_dy = np.diff(img, axis=1, append=np.expand_dims(img[:, -1], axis=1))

    elif operator == 'TF':
        recon_CG_dx = np.fft.ifft2(xi_dx.squeeze(0) * np.fft.fft2(recon_CG[-1], norm='ortho'), norm='ortho')
        recon_CG_dy = np.fft.ifft2(xi_dy.squeeze(0) * np.fft.fft2(recon_CG[-1], norm='ortho'), norm='ortho')
        img_dx = np.fft.ifft2(xi_dx.squeeze(0) * np.fft.fft2(img, norm='ortho'), norm='ortho')
        img_dy = np.fft.ifft2(xi_dy.squeeze(0) * np.fft.fft2(img, norm='ortho'), norm='ortho')

    elif operator == 'Convolution':
        recon_CG_dx = signal.convolve2d(recon_CG[-1], dx_kernel, mode='same', boundary='wrap')
        recon_CG_dy = signal.convolve2d(recon_CG[-1], dy_kernel, mode='same', boundary='wrap')
        img_dx = signal.convolve2d(img, dx_kernel, mode='same', boundary='wrap')
        img_dy = signal.convolve2d(img, dy_kernel, mode='same', boundary='wrap')

    mean_recon_CG_dx =  np.mean(np.abs(img_dx.real - recon_CG_dx.real) ** 2)
    mean_dxrecon_CG_with = np.mean(np.abs(img_dx.real - dxrecon_CG_with[-1].real) ** 2)
    mean_dxrecon_CG_without = np.mean(np.abs(img_dx.real - dxrecon_CG_without[-1].real) ** 2)

    mean_recon_CG_dy =  np.mean(np.abs(img_dy.real - recon_CG_dy.real) ** 2)
    mean_dyrecon_CG_with = np.mean(np.abs(img_dy.real - dyrecon_CG_with[-1].real) ** 2)
    mean_dyrecon_CG_without = np.mean(np.abs(img_dy.real - dyrecon_CG_without[-1].real) ** 2)

    ssim_recon_CG_dx = structural_similarity(img_dx.real, recon_CG_dx.real)
    ssim_dxrecon_CG_with = structural_similarity(img_dx.real, dxrecon_CG_with[-1].real)
    ssim_dxrecon_CG_without = structural_similarity(img_dx.real, dxrecon_CG_without[-1].real)

    ssim_recon_CG_dy = structural_similarity(img_dy.real, recon_CG_dy.real)
    ssim_dyrecon_CG_with = structural_similarity(img_dy.real, dyrecon_CG_with[-1].real)
    ssim_dyrecon_CG_without = structural_similarity(img_dy.real, dyrecon_CG_without[-1].real)

    plt.figure()
    plt.imshow(np.concatenate((img_dx.real, recon_CG_dx.real, dxrecon_CG_with[-1].real, dxrecon_CG_without[-1].real), axis=1), cmap='gray')
    plt.title('Dx (true, recon_CG, with, without)')
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.imshow(np.concatenate((img_dy.real, recon_CG_dy.real, dyrecon_CG_with[-1].real, dyrecon_CG_without[-1].real), axis=1), cmap='gray')
    plt.title('Dy (true, recon_CG, with, without)')
    plt.axis('off')
    plt.show()

    print('Mean')
    print(f'dx with {operator}')
    print('|| u_true - u_2steps ||           ', mean_recon_CG_dx)
    print('|| u_true - u_direct_incl ||       ', mean_dxrecon_CG_with)
    print('|| u_true - u_direct_omit ||    ', mean_dxrecon_CG_without, '\n')
    print(f'dy with {operator}')
    print('|| u_true - u_2steps ||           ', mean_recon_CG_dy)
    print('|| u_true - u_direct_incl ||       ', mean_dyrecon_CG_with)
    print('|| u_true - u_direct_omit ||    ', mean_dyrecon_CG_without, '\n')

    '''print('SSIM')
    print('img_dx, recon_CG_dx           ', ssim_recon_CG_dx)
    print('img_dx, dxrecon_CG_with       ', ssim_dxrecon_CG_with)
    print('img_dx, dxrecon_CG_without    ', ssim_dxrecon_CG_without, '\n')

    print('img_dy, recon_CG_dy           ', ssim_recon_CG_dy)
    print('img_dy, dyrecon_CG_with       ', ssim_dyrecon_CG_with)
    print('img_dy, dyrecon_CG_without    ', ssim_dyrecon_CG_without, '\n')'''


def correction_term_landweber(operator = 'Fourier', max_iter = 10, step_size = 1):
    img = np.load('../dataset/images/image_1007.npy').squeeze(2)
    mask = np.load('../dataset/sampling_pattern.npy').squeeze(2)
    coil = np.transpose(np.load('../dataset/coil_sensitivities/coil_sensitivities_1007.npy'), axes=[2, 0, 1]).astype(np.complex64)

    img /= np.max(img)
    g = np.fft.fft2(coil * img[np.newaxis, :, :], norm='ortho', axes=(1, 2))

    g *= mask
    if operator == 'FD':
        xi = np.exp(2 * 1j * np.pi * np.linspace(0, 191, 192) / 192) - 1
        xi_dx = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
        xi_dy = xi[np.newaxis, np.newaxis, :].astype(np.complex64)
    elif operator == 'TF':
        xi = 2 * np.pi * 1j * np.linspace(-96, 95, 192) / 192
        xi = np.fft.fftshift(xi)
        xi_dx = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
        xi_dy = xi[np.newaxis, np.newaxis, :].astype(np.complex64)
    elif operator =='Convolution':
        dx_kernel = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        dy_kernel = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        padded_dx_kernel = np.pad(dx_kernel, (0, 189), mode='constant')
        padded_dy_kernel = np.pad(dy_kernel, (0, 189), mode='constant')
        xi_dx = 192 * np.fft.fft2(padded_dx_kernel, norm='ortho')
        xi_dy = 192 * np.fft.fft2(padded_dy_kernel, norm='ortho')
    g_dx = xi_dx * g
    g_dy = xi_dy * g

    coil_dx = np.diff(coil, axis=1, append=np.expand_dims(coil[:, -1, :], axis=1))
    coil_dy = np.diff(coil, axis=2, append=np.expand_dims(coil[:, :, -1], axis=2))

    u_recon = [np.zeros((g.shape[1], g.shape[2])).astype(np.complex64)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coil * u_recon[-1], norm='ortho', axes=(-2, -1)) * mask
        u_recon.append(u_recon[-1] - step_size * np.sum(np.fft.ifft2((u1 - g) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))

    u_dxrecon = [np.zeros((g.shape[1], g.shape[2])).astype(np.complex64)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coil * u_dxrecon[-1], norm='ortho', axes=(-2, -1)) * mask
        if operator == 'FD':
            correction_dx = mask * np.fft.fft2(coil_dx * u_recon[i][np.newaxis, :, :], axes=(1, 2), norm='ortho')
            u_dxrecon.append(u_dxrecon[-1] - step_size * np.sum(np.fft.ifft2((u1 - g_dx + correction_dx) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))
        elif operator in ['TF','Convolution']:
            correction_dx = mask * np.fft.fft2(coil_dx * np.roll(u_recon[i], -1, axis=0)[np.newaxis, :, :], axes=(1, 2), norm='ortho')
            u_dxrecon.append(u_dxrecon[-1] - step_size * np.sum(np.fft.ifft2((u1 - g_dx + correction_dx) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))
        discrepancy_new = np.sqrt(np.sum(np.abs(g_dx - operator_A(u_dxrecon[-1], coil, mask))**2))

    u_dyrecon = [np.zeros((g.shape[1], g.shape[2])).astype(np.complex64)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coil * u_dyrecon[-1], norm='ortho', axes=(-2, -1)) * mask
        if operator == 'FD':
            correction_dy = mask * np.fft.fft2(coil_dy * u_recon[i][np.newaxis, :, :], axes=(1, 2), norm='ortho')
            u_dyrecon.append(u_dyrecon[-1] - step_size * np.sum(np.fft.ifft2((u1 - g_dy + correction_dy) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))
        elif operator in ['TF','Convolution']:
            correction_dy = mask * np.fft.fft2(coil_dy * np.roll(u_recon[i], -1, axis=1)[np.newaxis, :, :], axes=(1, 2), norm='ortho')
            u_dyrecon.append(u_dyrecon[-1] - step_size * np.sum(np.fft.ifft2((u1 - g_dy + correction_dy) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))

    u_dxrecon_short = [np.zeros((g.shape[1], g.shape[2])).astype(np.complex64)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coil * u_dxrecon_short[-1], norm='ortho', axes=(-2, -1)) * mask
        u_dxrecon_short.append(u_dxrecon_short[-1] - step_size * np.sum(np.fft.ifft2((u1 - g_dx) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))

    u_dyrecon_short = [np.zeros((g.shape[1], g.shape[2])).astype(np.complex64)]
    for i in range(max_iter):
        u1 = np.fft.fft2(coil * u_dyrecon_short[-1], norm='ortho', axes=(-2, -1)) * mask
        u_dyrecon_short.append(u_dyrecon_short[-1] - step_size * np.sum(np.fft.ifft2((u1 - g_dy) * mask, norm='ortho', axes=(-2, -1)) * np.conj(coil), axis=0))

    norm_x = np.sqrt(np.sum(np.abs(u_dxrecon_short[-1] - u_dxrecon[-1]) ** 2)) / np.sqrt(np.sum(np.abs(u_dxrecon[-1]) ** 2))
    norm_y = np.sqrt(np.sum(np.abs(u_dyrecon_short[-1] - u_dyrecon[-1]) ** 2)) / np.sqrt(np.sum(np.abs(u_dyrecon[-1]) ** 2))

    print(f'dx with {operator}')
    print('||Lu_incl - Lu_omit||/||Lu_incl||', norm_x, '\n')
    print(f'dy with {operator}')
    print('||Lu_incl - Lu_omit||/||Lu_incl||', norm_y, '\n')

    dx = np.concatenate([u_dxrecon[-1].real, u_dxrecon_short[-1].real], axis=1)
    dx -= np.min(dx)
    dx /= np.max(dx)
    dy = np.concatenate([u_dyrecon[-1].real, u_dyrecon_short[-1].real], axis=1)
    dy -= np.min(dy)
    dy /= np.max(dy)
    dx_dy = np.concatenate([u_dxrecon[-1].real + u_dyrecon[-1].real, u_dxrecon_short[-1].real + u_dyrecon_short[-1].real], axis=1)
    dx_dy -= np.min(dx_dy)
    dx_dy /= np.max(dx_dy)
    dx_dy_abs = np.concatenate([np.abs(u_dxrecon[-1].real) + np.abs(u_dyrecon[-1].real), np.abs(u_dxrecon_short[-1].real) + np.abs(u_dyrecon_short[-1].real)], axis=1)
    dx_dy_abs -= np.min(dx_dy_abs)
    dx_dy_abs /= np.max(dx_dy_abs)

    if operator == 'FD':
        img_dx = np.diff(img, axis=1, append=np.expand_dims(img[:, -1], axis=1))
        img_dy = np.diff(img, axis=0, append=np.expand_dims(img[-1,:], axis=0))
    elif operator == 'TF':
        img_dx = np.fft.ifft2(xi_dx.squeeze(0) * np.fft.fft2(img, norm='ortho'), norm='ortho')
        img_dy = np.fft.ifft2(xi_dy.squeeze(0) * np.fft.fft2(img, norm='ortho'), norm='ortho')
    elif operator == 'Convolution':
        img_dx = signal.convolve2d(img, dx_kernel, mode='same', boundary='wrap') #convolution2d(img, dx_kernel)
        img_dy = signal.convolve2d(img, dy_kernel, mode='same', boundary='wrap') #convolution2d(img, dy_kernel)

    mse_drecon_dx = np.mean((img_dx.real - u_dxrecon[-1].real)**2)
    mse_drecon_short_dx = np.mean((img_dx.real - u_dxrecon_short[-1].real)**2)
    mse_drecon_dy = np.mean((img_dy.real- u_dyrecon[-1].real)**2)
    mse_drecon_short_dy = np.mean((img_dy.real - u_dyrecon_short[-1].real)**2)

    ssim_drecon_dx = structural_similarity(img_dx.real, u_dxrecon[-1].real)
    ssim_drecon_short_dx = structural_similarity(img_dx.real, u_dxrecon_short[-1].real)
    ssim_drecon_dy = structural_similarity(img_dy.real, u_dyrecon[-1].real)
    ssim_drecon_short_dy = structural_similarity(img_dy.real, u_dyrecon_short[-1].real)

    print('\nMSE')
    print(f'dx with {operator}')
    print('|| u_true - u_direct_incl ||    ', mse_drecon_dx)
    print('|| u_true - u_direct_omit ||    ', mse_drecon_short_dx, '\n')
    print(f'dy with {operator}')
    print('|| u_true - u_direct_incl ||    ', mse_drecon_dy)
    print('|| u_true - u_direct_omit ||    ', mse_drecon_short_dy, '\n')

    '''print('SSIM')
    print('dx_with:     ', ssim_drecon_dx)
    print('dx_without:  ', ssim_drecon_short_dx)
    print('dy_with:     ', ssim_drecon_dy)
    print('dy_without:  ', ssim_drecon_short_dy, '\n')'''

if __name__ == "__main__":
    main()


