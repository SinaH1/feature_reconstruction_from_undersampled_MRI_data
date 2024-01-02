import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import loadmat
import copy
import adjusted_VarNet.mri_utils
from adjusted_VarNet.fft_utils import numpy_2_complex
import numpy as np
import os

DEFAULT_OPTS = {'root_dir': '../dataset/',
                'mri_data': 'simulated_mri',
                'coil_sens_data': 'coil_sensitivities',
                'true_image_data': 'images',
                'segmentation_data': 'segmentations',
                'mask': 'sampling_pattern.npy',
                'varnet_input_cg': 'varnet_input_cg',
                'varnet_input_landweber': 'varnet_input_landweber',
                'train_slices': [x for x in range(1, 2200)],
                'eval_slices': [x for x in range(2200, 2769)],
                'mode': 'train',
                'load_target': True,
                'normalization': 'max',
                'input': 'cg'
                }

class Mri_Segmentation_Dataset(Dataset):
    """ Dataset with simulated k-space data, coil sensitivities, true images and segmentations"""

    def __init__(self, **kwargs):
        """
        Parameters:
        root_dir: str
            root directory of data
        dataset_name: list of str
            list of directory to load data from
        transform:
        """
        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        self.root_dir = Path(self.options['root_dir'])
        self.mask_dir = self.root_dir / options['mask']
        self.mask = np.load(self.mask_dir).squeeze(2)

        # Processing directory
        print(self.root_dir / options['mri_data'])
        if not os.path.exists(self.root_dir / options['mri_data']):
            raise ValueError('Dataset not found!')
        if not os.path.exists(self.root_dir /options['coil_sens_data']):
            raise ValueError('Coil sensitivities not found!')
        if not os.path.exists(self.root_dir / options['true_image_data']):
            raise ValueError('True images not found!')
        if not os.path.exists(self.root_dir / options['segmentation_data']):
            raise ValueError('Segmentations not found!')

        self.filename = []
        self.coil_sens_list = []
        self.true_image_list = []
        self.varnet_input_list_cg = []
        self.varnet_input_list_landweber = []
        self.segmentation_list = []
        data_dir = self.root_dir / options['mri_data']
        coil_sens_dir = self.root_dir / options['coil_sens_data']
        true_image_dir = self.root_dir / options['true_image_data']
        varnet_input_cg_dir = self.root_dir / options['varnet_input_cg']
        varnet_input_landweber_dir = self.root_dir / options['varnet_input_landweber']
        segmentation_dir = self.root_dir / options['segmentation_data']

        # Load raw data and coil sensitivities name

        if options['mode'] == 'train':
            slice_no = options['train_slices']
        elif options['mode'] == 'eval_varnet' or options['mode'] == 'eval_seg':
            slice_no = options['eval_slices']

        for i in slice_no:
            slice_dir = data_dir / f'sim_mri_{i}.npy'
            self.filename.append(str(slice_dir))
            coil_sens_slice_dir = coil_sens_dir / f'coil_sensitivities_{i}.npy'
            self.coil_sens_list.append(str(coil_sens_slice_dir))
            true_image_slice_dir = true_image_dir / f'image_{i}.npy'
            self.true_image_list.append(str(true_image_slice_dir))
            varnet_input_cg_slice_dir = varnet_input_cg_dir / f'varnet_input_cg_{i}.npy'
            self.varnet_input_list_cg.append(str(varnet_input_cg_slice_dir))
            varnet_input_landweber_slice_dir = varnet_input_landweber_dir / f'varnet_input_landweber_{i}.npy'
            self.varnet_input_list_landweber.append(str(varnet_input_landweber_slice_dir))
            segmentation_slice_dir = segmentation_dir / f'segs_L_{i}.npy'
            self.segmentation_list.append(str(segmentation_slice_dir))

        self.mask_dir = self.root_dir / options['mask']
        print(self.mask_dir)
        self.mask = np.load(self.mask_dir).squeeze(2)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        mask = copy.deepcopy(self.mask)
        filename = self.filename[idx]
        coil_sens = self.coil_sens_list[idx]
        true_img = self.true_image_list[idx]
        varnet_input_cg = self.varnet_input_list_cg[idx]
        varnet_input_landweber = self.varnet_input_list_landweber[idx]
        seg_filename = self.segmentation_list[idx]

        raw_data = np.transpose(np.load(filename), axes=[2, 0, 1])
        f = np.ascontiguousarray(raw_data).astype(np.complex64)

        coil_sens_data = np.transpose(np.load(coil_sens), axes=[2, 0, 1])
        c = np.ascontiguousarray(coil_sens_data).astype(np.complex64)

        seg = np.load(seg_filename)
        seg = seg.reshape((1, 192, 192))

        if self.options['load_target']:
            ref = np.load(true_img).squeeze(2).astype(np.complex64)
        else:
            ref = np.zeros_like(mask, dtype=np.complex64)

        if self.options['input'] == 'ifft':
            # mask rawdata
            f = np.multiply(mask, f)
            # compute initial image input
            input0 = np.sum(np.fft.ifft2(f, norm='ortho'), axis=0).astype(np.complex64)
        elif self.options['input'] == 'cg':
            input0 = np.load(varnet_input_cg).astype(np.complex64)
        elif self.options['input'] == 'landweber':
            input0 = np.load(varnet_input_landweber).astype(np.complex64)
        elif self.options['input'] == 'adjoint':
            f = np.multiply(mask, f)
            input0 = mri_utils.mriAdjointOp(f, c, mask).astype(np.complex64)
        # print('___input0:', input0.shape)


        # normalize the data
        if self.options['normalization'] == 'max':
            norm = np.max(np.abs(input0))
            norm_ref = np.max(np.abs(ref))
        elif self.options['normalization'] == 'no':
            norm = 1.0
            norm_ref = 1.0
        else:
            raise ValueError("Normalization has to be in ['max','no']")
        
        f /= norm
        input0 /= norm
        
        if self.options['load_target']:
            ref /= norm_ref
        else:
            ref = np.zeros_like(input0)

        input0 = numpy_2_complex(input0)
        f = numpy_2_complex(f)
        c = numpy_2_complex(c)
        mask = torch.from_numpy(mask)
        ref = numpy_2_complex(ref)
        seg[seg > 0] = 1

        # print('_______input0',input0.size()) # HxWx2
        # print('_______f',f.size()‚) # 4xHxWx2
        # print('_______c',c.size()) # 4xHxWx2
        # print('_______mask',mask.size()) # HxW
        # print('_______ref',ref.size()) # HxWx2
        data = {'u_t': input0, 'f': f, 'coil_sens': c, 'sampling_mask': mask, 'reference': ref, 'seg': seg}
        return data






