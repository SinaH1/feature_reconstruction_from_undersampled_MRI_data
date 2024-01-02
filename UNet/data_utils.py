import sys
sys.path.insert(0,'../')
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import copy
import numpy as np
import os
from scipy import signal


DEFAULT_OPTS = {'root_dir': '../datasets',
                'image_data': 'images',
                'segmentation_data': 'segmentations',
                'train_slices': [x for x in range(1, 2200)], # 1,1000 
                'eval_slices': [x for x in range(2200, 2769)], # 1000,1282
                'mode': 'train',
                'load_target': True,
                'normalization': 'max'
                }


class Torso_Dataset(Dataset):
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

        # Processing directory
        if not os.path.exists(self.root_dir / options['image_data']):
            raise ValueError('Dataset not found!')
        if not os.path.exists(self.root_dir / options['segmentation_data']):
            raise ValueError('Segmentations not found!')

        self.filename_list = []
        self.segmentation_list = []
        data_dir = self.root_dir / options['image_data']
        segmentation_dir = self.root_dir / options['segmentation_data']

        # Load raw data and coil sensitivities name
        if options['mode'] == 'train':
            slice_no = options['train_slices']
        elif options['mode'] == 'eval':
            slice_no = options['eval_slices']

        for i in slice_no:
            slice_dir = data_dir / f'image_{i}.npy'
            self.filename_list.append(str(slice_dir))
            segmentation_slice_dir = segmentation_dir / f'segs_L_{i}.npy'
            self.segmentation_list.append(str(segmentation_slice_dir))

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        segs_filename = self.segmentation_list[idx]

        torso_data = np.load(filename)
        segs = np.load(segs_filename)
        if self.options['net_type'] =='orig+dx':
            torso_data_dx = np.diff(torso_data.squeeze(2), axis=0, append=np.expand_dims(torso_data.squeeze(2)[-1,:], axis=0))
            torso_data = np.concatenate((torso_data.reshape((1,192,192)).astype(float), np.expand_dims(torso_data_dx, axis=0)), axis=0)

        elif self.options['net_type'] == 'learned_kernels':
            kernel = 2*np.random.rand(3,3)-1
            torso_data = signal.convolve2d(torso_data.squeeze(-1), kernel, mode='same', boundary='wrap', fillvalue=0)#.astype(np.complex64)            
            torso_data = np.expand_dims(torso_data, axis=0)#.astype(np.float64)
        
        elif self.options['net_type'] == 'orig+kernel':
            kernel = 2*np.random.rand(3,3)-1
            torso_data_kernel = signal.convolve2d(torso_data.squeeze(-1), kernel, mode='same', boundary='wrap', fillvalue=0)#.astype(np.complex64)            
            torso_data_kernel = np.expand_dims(torso_data_kernel, axis=0)#.astype(np.float64)
            torso_data = np.concatenate((torso_data.reshape((1,192,192)).astype(float),torso_data_kernel), axis=0)

        # normalize the data
        if self.options['normalization'] == 'max':
            norm = np.max(np.abs(torso_data))
        elif self.options['normalization'] == 'no':
            norm = 1.0
        else:
            raise ValueError("Normalization has to be in ['max','no']")

        torso_data /= norm
        segs[segs > 0] = 1
        
        data = {'img': torso_data.reshape((1,192,192)).astype(float), 'segs': segs.reshape((1,192,192)).astype(float)}
        return data















