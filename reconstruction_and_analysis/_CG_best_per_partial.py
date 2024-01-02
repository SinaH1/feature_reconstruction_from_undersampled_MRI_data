import sys
sys.path.insert(0, '../')
import os
from skimage.metrics import structural_similarity
from utils import *
import argparse

parser = argparse.ArgumentParser(description='CG arguments')
parser.add_argument('--operator', type=str, default='TF',
                        help='Operator for partial derivative of image. Choices are: TF (Truncated Fourier series), FD (Forward Differences), Convolution')
parser.add_argument('--show_images', type=bool, default=False,
                        help='show reconstructed images')

def main():
    args = parser.parse_args()
    args = vars(args)
    operator = args['operator']
    show_images = args['show_images']
    test_several_noise(operator = operator, show_images=show_images)

def test_several_noise(operator='FD', show_images=False):
    range_noise = 21
    noise = np.linspace(0,2e-1,range_noise)
    mse_direct_best = np.zeros(range_noise)
    mse_2steps_best = np.zeros(range_noise)
    ssim_direct_best = np.zeros(range_noise)
    ssim_2steps_best = np.zeros(range_noise)

    use_reference = operator
    use_exact = False

    if use_exact:
        if not os.path.exists(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/per_std'):
            os.makedirs(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/per_std')
        if not os.path.exists(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/images'):
            os.makedirs(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/images')

    else:
        if not os.path.exists(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/per_std'):
            os.makedirs(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/per_std')
        if not os.path.exists(f'../results/edge_detection/CG_{use_reference}_per_partial/images'):
            os.makedirs(f'../results/edge_detection/CG_{use_reference}_per_partial/images')


    for i in range(range_noise):
        data = reconstruction_landweber(noise_std=noise[i], max_iter=30, show_images=show_images, plot_mse=True, use_reference=use_reference, use_exact=use_exact)

        mse_direct_best[i] = data['MSE_direct'][1]
        mse_2steps_best[i] = data['MSE_2steps'][1]
        ssim_direct_best[i] = data['SSIM_direct'][1]
        ssim_2steps_best[i] = data['SSIM_2steps'][1]

    if use_exact:
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/mse_direct_{use_reference}_best.npy',mse_direct_best)
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/mse_2steps_{use_reference}_best.npy', mse_2steps_best)
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/ssim_direct_{use_reference}_best.npy', ssim_direct_best)
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/ssim_2steps_{use_reference}_best.npy', ssim_2steps_best)
    else:
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/mse_direct_{use_reference}_best.npy',mse_direct_best)
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/mse_2steps_{use_reference}_best.npy',mse_2steps_best)
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/ssim_direct_{use_reference}_best.npy',ssim_direct_best)
        np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/ssim_2steps_{use_reference}_best.npy',ssim_2steps_best)

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle(f'Ableitung {use_reference}')
    #axs[0].plot(noise, mse_direct_short_best, label='dxrecon_short')
    axs[0].plot(noise, mse_direct_best, label = 'direct')
    axs[0].plot(noise, mse_2steps_best, label='2 steps')
    axs[0].legend(loc=1)
    axs[0].set_ylabel('MSE')
    #axs[2].plot(noise, ssim_direct_short_best, label='dxrecon_short')
    axs[1].plot(noise, ssim_direct_best, label='direct')
    axs[1].plot(noise, ssim_2steps_best, label='2 steps')
    axs[1].set_ylabel('SSIM')
    axs[1].set_xlabel('noise std')
    if use_exact:
        plt.savefig(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/CG_{use_reference}_MSE_SSIM.png')
    else:
        plt.savefig(f'../results/edge_detection/CG_{use_reference}_per_partial/CG_{use_reference}_MSE_SSIM.png')
    plt.show()

def reconstruction_landweber(noise_std = 1e-1, max_iter = 50, show_images = True, plot_mse = True, use_reference='FD', use_exact=False):
    print('\n--------------------------------------')
    print('noise std:           ', noise_std)
    print('maximal iterations:  ', max_iter)
    print('reference:           ', use_reference)
    print('exact:               ', use_exact)

    input_path = '../dataset'
    mask = np.load('../dataset/sampling_pattern.npy').squeeze(2)
    # amount_of_data = len([entry for entry in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, entry))])
    amount_of_data = 1
    step_size = 1

    start = 1007
    for i in range(start, start + amount_of_data):
        img = np.load(input_path + f'/images/image_{i + 1}.npy').squeeze(2).astype(np.complex64)  # HxW
        coil = np.transpose(np.load(f'../dataset/coil_sensitivities/coil_sensitivities_{i + 1}.npy'), axes=[2, 0, 1]).astype(np.complex64)  # CxHxW
        f = np.fft.fft2(coil * img[np.newaxis, :, :], norm='ortho', axes=(1, 2))

        gauss = np.zeros_like(f)
        gauss.real = np.random.normal(0, noise_std * f.real.std(), f.shape)
        gauss.imag = np.random.normal(0, noise_std * f.imag.std(), f.shape)
        f = f + gauss  # noisy data
        f = mask * f

        # Calculate
        if use_reference == 'FD':
            xi = np.exp(2 * np.pi * 1j * np.linspace(0, 191, 192) / 192) - 1
            xi_dx = xi[np.newaxis, np.newaxis, :].astype(np.complex64)
            xi_dy = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
            f_dx = xi_dx * f
            f_dy = xi_dy * f
            img_dx = np.diff(img, append=np.expand_dims(img[:, -1], axis=1), axis=1)
            img_dy = np.diff(img, append=np.expand_dims(img[-1, :], axis=0), axis=0)
            img_fd = np.abs(img_dx) + np.abs(img_dy)
        elif use_reference == 'TF':
            xi = 2 * np.pi * 1j * np.linspace(-96, 95, 192)/192
            xi = np.fft.fftshift(xi)
            xi_dx = xi[np.newaxis, np.newaxis, :].astype(np.complex64)
            xi_dy = xi[np.newaxis, :, np.newaxis].astype(np.complex64)
            f_dx = xi_dx * f
            f_dy = xi_dy * f
            img_dx = np.fft.ifft2(xi_dx.squeeze(0) * np.fft.fft2(img, norm='ortho'), norm='ortho')
            img_dy = np.fft.ifft2(xi_dy.squeeze(0)  * np.fft.fft2(img, norm='ortho'), norm='ortho')
            img_fd = np.abs(img_dx.real) + np.abs(img_dy.real)
        else:
            raise ValueError('Choose use_reference between FD and TF.')


        # u = np.sum(np.fft.ifft2(f * mask, axes=axes) * np.conj(coils), axis=0)
        u_recon = CGNE_plot(f, coil, mask, max_iter=max_iter)
        u_recon = u_recon[1:]
        u_recon_dx = []
        u_recon_dy = []
        mse_orig_dx = np.zeros(max_iter)
        mse_orig_dy = np.zeros(max_iter)
        ssim_orig_dx = np.zeros(max_iter)
        ssim_orig_dy = np.zeros(max_iter)
        for i in range(max_iter):
            if use_reference == 'FD':
                u_recon_dx.append(np.diff(u_recon[i], append=np.expand_dims(img[:, -1], axis=1), axis=1))
                u_recon_dy.append(np.diff(u_recon[i], append=np.expand_dims(img[-1, :], axis=0), axis=0))
            elif use_reference == 'TF':
                u_recon_dx.append(derivative_fourier(xi_dx.squeeze(0), u_recon[i]))
                u_recon_dy.append(derivative_fourier(xi_dy.squeeze(0), u_recon[i]))
            mse_orig_dx[i] = mse(img_dx.real, u_recon_dx[i].real)
            mse_orig_dy[i] = mse(img_dy.real, u_recon_dy[i].real)
            ssim_orig_dx[i] = structural_similarity(img_dx.real.astype(np.float64), u_recon_dx[i].real.astype(np.float64))
            ssim_orig_dy[i] = structural_similarity(img_dy.real.astype(np.float64), u_recon_dy[i].real.astype(np.float64))

        u_recon_d_mse = np.abs(u_recon_dx[np.argmin(mse_orig_dx)].real) + np.abs(u_recon_dy[np.argmin(mse_orig_dy)].real)
        u_recon_d_ssim = np.abs(u_recon_dx[np.argmax(ssim_orig_dx)].real) + np.abs(u_recon_dy[np.argmax(ssim_orig_dy)].real)

        mse_orig = mse(img_fd, u_recon_d_mse)
        ssim_orig = structural_similarity(img_fd, u_recon_d_ssim)

        if use_exact:
            coil_dx = np.diff(coil, axis=2, append=np.expand_dims(coil[:, :, -1], axis=2))
            coil_dy = np.diff(coil, axis=1, append=np.expand_dims(coil[:, -1, :], axis=1))

            if use_reference == 'TF':
                correction_dx = mask * np.fft.fft2(coil_dx * u_recon[-1][np.newaxis, :, :], axes=(1, 2), norm='ortho')
                correction_dy = mask * np.fft.fft2(coil_dy * u_recon[-1][np.newaxis, :, :], axes=(1, 2), norm='ortho')

            elif use_reference == 'FD':
                correction_dx = mask * np.fft.fft2(coil_dx * np.roll(u_recon[-1], -1, axis=1)[np.newaxis, :, :], axes=(1, 2), norm='ortho')
                correction_dy = mask * np.fft.fft2(coil_dy * np.roll(u_recon[-1], -1, axis=0)[np.newaxis, :, :], axes=(1, 2), norm='ortho')

        if use_exact:
            u_dxrecon = CGNE_plot(f_dx - correction_dx, coil, mask, max_iter=max_iter)
            u_dyrecon = CGNE_plot(f_dy - correction_dy, coil, mask, max_iter=max_iter)
        else:
            u_dxrecon = CGNE_plot(f_dx, coil, mask, max_iter=max_iter)
            u_dyrecon = CGNE_plot(f_dy, coil, mask, max_iter=max_iter)
        u_dxrecon = u_dxrecon[1:]
        u_dyrecon = u_dyrecon[1:]
        mse_drecon_x      = np.zeros(max_iter)
        ssim_drecon_x     = np.zeros(max_iter)
        mse_drecon_y = np.zeros(max_iter)
        ssim_drecon_y = np.zeros(max_iter)

        for i in range(max_iter):
            mse_drecon_x[i] = mse(img_dx.real, u_dxrecon[i].real)
            mse_drecon_y[i] = mse(img_dy.real, u_dyrecon[i].real)
            ssim_drecon_x[i] = structural_similarity(img_dx.real, u_dxrecon[i].real)
            ssim_drecon_y[i] = structural_similarity(img_dy.real, u_dyrecon[i].real)

        mse_drecon  = mse(img_fd, np.abs(u_dxrecon[np.argmin(mse_drecon_x)].real) + np.abs(u_dyrecon[np.argmin(mse_drecon_y)].real))
        ssim_drecon  = structural_similarity(img_fd, np.abs(u_dxrecon[np.argmax(ssim_drecon_x)].real) + np.abs(u_dyrecon[np.argmax(ssim_drecon_y)].real))


        results = {'MSE_direct': [noise_std, mse_drecon],
                   'MSE_2steps': [noise_std, mse_orig],
                   'SSIM_direct': [noise_std, ssim_drecon],
                   'SSIM_2steps': [noise_std, ssim_orig]
                   }

        print('Best reconstructions:')
        print(f'MSE  - direct : {mse_drecon}')
        print(f'MSE  - 2steps : {mse_orig}\n')
        print(f'SSIM - direct : {ssim_drecon}')
        print(f'SSIM - 2steps : {ssim_orig}')

        if show_images:
            fig = plt.figure()
            gs = fig.add_gridspec(1,3, wspace=0)
            axs = gs.subplots(sharex=True)
            fig.suptitle('Std {:.4f} with {}'.format(noise_std,use_reference))
            axs[0].imshow((255*(np.abs(u_dxrecon[np.argmin(mse_drecon)]) + np.abs(u_dyrecon[np.argmin(mse_drecon)]))).astype(int), cmap='gray', interpolation='nearest')
            axs[0].set_title(f'Direct at {np.max([5,np.argmin(mse_drecon)])}' )
            axs[0].axis('off')
            axs[1].imshow((255*u_recon_d_mse).astype(int), cmap='gray', interpolation='nearest')
            axs[1].set_title(f'2 Steps at {np.max([5, np.argmin(mse_orig)])}')
            axs[1].axis('off')
            axs[2].imshow((255*np.abs(img_fd)).astype(int), cmap='gray', interpolation='nearest')
            axs[2].set_title(f'Original')
            axs[2].axis('off')
            plt.show()

            show_img(np.concatenate([np.abs(u_dxrecon[np.argmin(mse_drecon_x)].real) + np.abs(u_dyrecon[np.argmin(mse_drecon_y)].real), u_recon_d_mse.real, img_fd], axis=1), title = 'Std {:.4f} with Landweber'.format(noise_std))
        imgs_recon = np.concatenate([np.abs(u_dxrecon[np.argmin(mse_drecon_x)].real) + np.abs(u_dyrecon[np.argmin(mse_drecon_y)].real), u_recon_d_mse.real, img_fd], axis=1)
        imgs_recon /=np.max(imgs_recon)
        if use_exact:
            plt.imsave(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/images/recons_CG_best_{use_reference}_{noise_std}.png', (255.0 * imgs_recon).astype(np.uint8), cmap='gray')
        else:
            plt.imsave(f'../results/edge_detection/CG_{use_reference}_per_partial/images/recons_CG_best_{use_reference}_{noise_std}.png',(255.0 * imgs_recon).astype(np.uint8), cmap='gray')

        if noise_std in [1e-2,1e-1,2e-1]:
            if use_exact:
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/per_std/mse_2steps_{use_reference}_{noise_std}.npy', mse_orig)
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/per_std/mse_direct_{use_reference}_{noise_std}.npy', mse_drecon)
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/per_std/ssim_2steps_{use_reference}_{noise_std}.npy', ssim_orig)
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial_exact/npy_files/per_std/ssim_direct_{use_reference}_{noise_std}.npy', ssim_drecon)
            else:
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/per_std/mse_2steps_{use_reference}_{noise_std}.npy', mse_orig)
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/per_std/mse_direct_{use_reference}_{noise_std}.npy', mse_drecon)
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/per_std/ssim_2steps_{use_reference}_{noise_std}.npy', ssim_orig)
                np.save(f'../results/edge_detection/CG_{use_reference}_per_partial/npy_files/per_std/ssim_direct_{use_reference}_{noise_std}.npy', ssim_drecon)
        return results

if __name__ == "__main__":
    main()
