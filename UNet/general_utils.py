from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import imageio

def print_options(parser, opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)
    file_name =  save_dir / '{}_opt.txt'.format(opt.mode)
    with open(str(file_name), 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def save_loss_as_plot(dir, losses, val_losses=None):
    #plt.clf()
    plt.switch_backend('agg')
    plt.plot(np.arange(len(losses)), losses, label='train')
    if val_losses is not None:
        plt.plot(np.arange(len(val_losses)), val_losses, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.legend()
    plt.savefig(dir)

    
def save_result_seg_img(save_dir, true_img, true_seg, seg):
    true_img = true_img.detach().cpu().numpy()
    true_seg = true_seg.detach().cpu().numpy()
    seg = seg.detach().cpu().numpy()
    diff = np.abs(true_seg - seg)
    img_to_save = np.concatenate((true_img, true_seg, seg, diff), axis=1)
    img_to_save = 255*(img_to_save/np.max(img_to_save))
    imageio.imwrite(str(save_dir), img_to_save.astype(np.uint8))
    
