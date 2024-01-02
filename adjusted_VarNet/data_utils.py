import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import signal
import copy
import mri_utils
from fft_utils import numpy_2_complex
import numpy as np
import os


DEFAULT_OPTS = {'root_dir':'../dataset/',
				'mri_data': 'simulated_mri',
				'coil_sens_data': 'coil_sensitivities',
				'true_image_data': 'images',
				'mask': 'sampling_pattern.npy',
				'varnet_input_cg': 'varnet_input_cg',
				'varnet_input_landweber': 'varnet_input_landweber',
				'train_slices': [x for x in range(1,1000)], #  1, 2200			-> 1-1000 for subset, 1-2200 for whole dataset
				'eval_slices': [x for x in range(1000,1282)], #  2200, 2769 	-> 1000-1282 for subset, 2200-2769 for whole dataset
				'mode':'train',
				'load_target':True,
				'normalization':'max',
				'input': 'cg'
}

class Simulated_Dataset(Dataset):
	""" Simulated MRI knee dataset with simulated k-space data, coil sensitivities and true images"""
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
		self.mask_dir = self.root_dir/ options['mask']
		self.mask = np.load(self.mask_dir).squeeze(2)

		# Processing directory
		print(self.root_dir / options['mri_data'])
		if not os.path.exists(self.root_dir / options['mri_data']):
			raise ValueError('Dataset not found!')
		if not os.path.exists(self.root_dir / options['coil_sens_data']):
			raise ValueError('Coil sensitivities not found!')
		if not os.path.exists(self.root_dir / options['true_image_data']):
			raise ValueError('True images not found!')

		self.filename = []
		self.coil_sens_list = []
		self.true_image_list = []
		self.varnet_input_list_cg = []
		self.varnet_input_list_landweber = []
		data_dir = self.root_dir / options['mri_data']
		coil_sens_dir = self.root_dir / options['coil_sens_data']
		true_image_dir = self.root_dir/ options['true_image_data']
		varnet_input_cg_dir = self.root_dir/ options['varnet_input_cg']
		varnet_input_landweber_dir = self.root_dir/ options['varnet_input_landweber']

		# Load raw data and coil sensitivities name
		if options['mode'] == 'train':
			slice_no = options['train_slices']
		elif options['mode'] in ['eval', 'eval_image_per_cascade']:
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

		self.mask_dir = self.root_dir/ options['mask']
		self.mask = np.load(self.mask_dir).squeeze(2)

	def __len__(self):
		return len(self.filename)

	def __getitem__(self,idx):
		mask = copy.deepcopy(self.mask)
		filename = self.filename[idx]
		coil_sens = self.coil_sens_list[idx]
		true_img = self.true_image_list[idx]
		varnet_input_cg = self.varnet_input_list_cg[idx]
		varnet_input_landweber = self.varnet_input_list_landweber[idx]

		raw_data = np.transpose(np.load(filename), axes=[2, 0, 1])
		f = np.ascontiguousarray(raw_data).astype(np.complex64)
		
		coil_sens_data = np.transpose(np.load(coil_sens), axes=[2, 0, 1])
		c = np.ascontiguousarray(coil_sens_data).astype(np.complex64)

		if self.options['load_target']:
			ref = np.load(true_img).squeeze(2).astype(np.complex64)
		else:
			ref = np.zeros_like(mask,dtype=np.complex64)

		if self.options['input']=='ifft':
			# mask rawdata
			f = np.multiply(mask, f)
			# compute initial image input
			input0 = np.sum(np.fft.ifft2(f, norm='ortho'), axis=0).astype(np.complex64)
		elif self.options['input']=='cg': 
			input0 = np.load(varnet_input_cg).astype(np.complex64)
		elif self.options['input']=='landweber': 
			input0 = np.load(varnet_input_landweber).astype(np.complex64)
		elif self.options['input']=='adjoint':
			f = np.multiply(mask, f)
			input0 = mri_utils.mriAdjointOp(f,c,mask).astype(np.complex64)

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

		if self.options['operator'] == 'TF':
			xi = np.fft.fftshift(2 * 1j * np.pi * np.linspace(-96, 95, ref.shape[0]) / ref.shape[0])
		elif self.options['operator'] == 'FD':
			xi = np.exp(2 * np.pi * 1j * np.linspace(0, ref.shape[0]-1, ref.shape[0]) / ref.shape[0]) - 1
		
		if self.options['net_type'] == 'only_dx':
			xi_dx = xi[np.newaxis, :, np.newaxis]
			f *= xi_dx
			if self.options['operator'] == 'FD':
				input0 = np.diff(input0, axis=0, append=np.expand_dims(input0[0,:], axis=0)).astype(np.complex64)
				ref = np.diff(ref, axis=0, append=np.expand_dims(ref[0,:], axis=0)).astype(np.complex64)
			elif self.options['operator'] == 'TF':
				input0 = np.fft.ifft2(xi_dx * np.fft.fft2(input0, norm='ortho'), norm='ortho').astype(np.complex64).squeeze(0)
				ref = np.fft.ifft2(xi_dx * np.fft.fft2(ref, norm='ortho'), norm='ortho').astype(np.complex64).squeeze(0)
		
		elif self.options['net_type'] == 'only_dy':
			xi_dy = xi[np.newaxis, np.newaxis,:]
			f *= xi_dy
			if self.options['operator'] == 'FD':
				input0 = np.diff(input0, axis=1, append=np.expand_dims(input0[:,0], axis=1)).astype(np.complex64)
				ref = np.diff(ref, axis=1, append=np.expand_dims(ref[:,0], axis=1)).astype(np.complex64)
			elif self.options['operator'] == 'TF':
				input0 = np.fft.ifft2(xi_dy * np.fft.fft2(input0, norm='ortho'), norm='ortho').astype(np.complex64).squeeze(0)
				ref = np.fft.ifft2(xi_dy * np.fft.fft2(ref, norm='ortho'), norm='ortho').astype(np.complex64).squeeze(0)

		elif self.options['net_type'] == 'learned_kernels':
			kernel = 2*np.random.rand(3,3)-1
			kernel_pad = np.pad(kernel, (0, 189), mode='constant')
			f *= np.fft.fft2(kernel_pad)[np.newaxis,...].astype(np.complex64)
			ref = signal.convolve2d(ref, kernel, mode='same', boundary='wrap', fillvalue=0).astype(np.complex64)
			input0 = signal.convolve2d(input0, kernel, mode='same', boundary='wrap', fillvalue=0).astype(np.complex64)

		input0 = numpy_2_complex(input0)
		f = numpy_2_complex(f)
		c = numpy_2_complex(c)
		mask = torch.from_numpy(mask)
		ref = numpy_2_complex(ref)

		data = {'u_t': input0, 'f': f, 'coil_sens': c, 'sampling_mask': mask, 'reference': ref}
		return data


