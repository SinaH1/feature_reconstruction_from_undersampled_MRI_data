import sys
sys.path.insert(0,'../')
from adjusted_VarNet.models import VariationalNetwork
from adjusted_VarNet.models_learned_kernels import VariationalNetwork_learned_kernels
import argparse
from adjusted_VarNet.data_utils import *
from torch.utils.data import DataLoader
from pathlib import Path
from adjusted_VarNet.misc_utils import print_options, save_loss_as_plot
import tqdm

import os
parser = argparse.ArgumentParser(description='Variational network arguments')

# Data IO
parser.add_argument('--name',             type=str, default='simulated_mri',     help='name of the dataset to use')
parser.add_argument('--root_dir',         type=str, default='../dataset/',         help='directory of the data')

# Network configuration
parser.add_argument ('--net_type',    	type=str, default='original', help='original, only_dx, only_dy, random_kernels, fixed_kernels')
parser.add_argument ('--operator',    	type=str, default='TF', help='FD,TF')
parser.add_argument('--features_out',     type=int, default=48,    help='number of filter for convolutional kernel')
parser.add_argument('--num_act_weights',  type=int, default=31,    help='number of RBF kernel for activation function')
parser.add_argument('--num_stages',       type=int, default=10,    help='number of stages in the network')
parser.add_argument('--activation',       type=str, default='rbf', help='activation function to use (rbf or relu)')

# Training and Testing Configuration
parser.add_argument('--mode',             type=str,   default='train',               help='train,eval, eval_image_per_cascade')
parser.add_argument('--optimizer',        type=str,   default='adam',                help='type of optimizer to use for training')
parser.add_argument('--loss_type',        type=str,   default='complex',             help='compute loss on complex or magnitude image or absolute or edges')
parser.add_argument('--lr',               type=float, default=1e-4,                  help='learning rate')
parser.add_argument('--epoch',            type=int,   default=100,                   help='number of training epoch')
parser.add_argument('--batch_size',       type=int,   default=16,                     help='batch size')
parser.add_argument('--gpus',             type=str,   default='0',        	         help='gpu id to use')
parser.add_argument('--save_dir',         type=str,   default='../results/adjusted_VarNet/test',    help='directory of the experiment')
parser.add_argument('--pretrained_model', type=str,   default=None) #'/usr/local/ssd2t/heizmasa/Masterarbeit/results/adjusted_VarNet/_varnet_original_MSE/varnet_best.h5', help='path to pretrained model')
parser.add_argument('--momentum',         type=float, default=0.,                    help='momentum for the optimizer')
parser.add_argument('--loss_weight',      type=float, default=1.,                    help='weight for the loss function')
parser.add_argument('--error_scale',      type=float, default=1.,                    help='how much to magnify the error map for display purpose')
parser.add_argument('--save_loss_plot',   type=bool, default=True,                    help='save plot of mean loss per epoch')
parser.add_argument('--save_mean_loss',   type=bool, default=True,                    help='save mean loss per epoch as numpy array')
parser.add_argument('--input',            type=str, default='cg', help='input for VarNet: ifft, cg(10 iterations), landweber(50 iterations)')
parser.add_argument('--check_validation_per_epoch', type=bool, default=True, help='Evaluate validation set per epoch')


def main():
	args = parser.parse_args()
	print_options(parser,args)
	args = vars(args)

	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	args['device'] = device
	# setting up network
	if args['net_type'] in ['original', 'only_dx','only_dy']: #or args['net_type'] == 'only_dx':
		varnet = VariationalNetwork(**args)
	elif args['net_type'] in ['random_kernels','fixed_kernels']:
		args['image_size'] = [192,192]
		varnet = VariationalNetwork_learned_kernels(**args)

	varnet.to(device)
	pytorch_total_params = sum(p.numel() for p in varnet.parameters() if p.requires_grad)
	print('Total number of parameters:', pytorch_total_params)

	# setting up data loader
	dataset = Simulated_Dataset(**args)
	# start training
	save_dir = Path(args['save_dir'])
	save_dir.mkdir(parents=True,exist_ok=True)

	if args['mode'] == 'train':
		shuffle = True
		if args['check_validation_per_epoch']:
			best_val_loss = float('inf')
			validation_split = .25
			random_seed= 42
		
			# Creating data indices for training and validation splits:
			dataset_size = len(dataset)
			validation_size = int(np.floor(validation_split * dataset_size))
			training_size = dataset_size - validation_size
			train_dataset, val_dataset = torch.utils.data.random_split(dataset, [training_size, validation_size], torch.Generator().manual_seed(random_seed))

			train_dataloader = DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=shuffle, pin_memory=True)
			val_dataloader = DataLoader(val_dataset,batch_size=args['batch_size'],shuffle=False, pin_memory=True)   
		else: 
			train_dataloader = DataLoader(dataset,batch_size=args['batch_size'],shuffle=shuffle,pin_memory=True)

		if args['pretrained_model']:
				varnet.load_state_dict(torch.load(str(args['pretrained_model'])))
		torch.set_grad_enabled(True)
		losses_per_epoch = []
		val_losses_per_epoch = []
		optimizer = varnet.configure_optimizers()
		for epoch in range(args['epoch']):
			varnet.train()
			data_dqdm_loop = tqdm.tqdm(train_dataloader)
			losses = []
			for batch_idx, batch in enumerate(data_dqdm_loop):
				data_dqdm_loop.set_description(f"Epoch {epoch}")
				# train step
				batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
				logs = varnet.training_step(batch, batch_idx)

				# clear gradients
				if args['optimizer']!='iipg':
					optimizer.zero_grad()
				# backward
				logs['loss'].backward()
				# update parameters
				optimizer.step()
				losses.append(logs['loss'].item())
				if epoch==0:
					data_dqdm_loop.set_postfix(loss=logs['loss'].item())
				else:
					data_dqdm_loop.set_postfix(loss=logs['loss'].item(), epoch_loss=losses_per_epoch[-1])
				
			if args['check_validation_per_epoch']:
				varnet.eval()
				val_losses = []
				data_dqdm_loop = tqdm.tqdm(val_dataloader)
				for batch_idx, batch in enumerate(data_dqdm_loop):
					data_dqdm_loop.set_description(f"Validation {epoch}")
					batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
					logs_val = varnet.test_step(batch, batch_idx, save=False)
					val_losses.append(logs_val['test_loss'])
					if epoch==0:
						data_dqdm_loop.set_postfix(loss=logs_val['test_loss'])
					else:
						data_dqdm_loop.set_postfix(loss=logs_val['test_loss'], epoch_loss=val_losses_per_epoch[-1])
				val_losses_per_epoch.append(sum(val_losses) / len(val_losses))
				if args['save_mean_loss']:
					np.save(str(save_dir / 'val_losses_per_epoch'), val_losses_per_epoch)
				if val_losses_per_epoch[-1] < best_val_loss:
					best_val_loss = val_losses_per_epoch[-1]
					torch.save(varnet.state_dict(), str(save_dir / 'varnet_best.h5'))
					with open(save_dir / 'train_opt.txt', 'r+') as train_txt_file:
						lines = train_txt_file.readlines()
						lines[-1] = f'best validation loss: {best_val_loss} at epoch {epoch}'
						train_txt_file.seek(0)
						train_txt_file.writelines(lines)

			losses_per_epoch.append(sum(losses) / len(losses))
			data_dqdm_loop.set_description("Loss per epoch: {}".format(losses_per_epoch[-1]))
			if args['save_mean_loss']:
				np.save(str(save_dir / 'losses_per_epoch'), losses_per_epoch)
			if args['save_loss_plot']:
				if args['check_validation_per_epoch']:
					save_loss_as_plot(str(save_dir / 'losses_per_epoch.png'), losses_per_epoch, val_losses_per_epoch)
				else:
					save_loss_as_plot(str(save_dir / 'losses_per_epoch.png'), losses_per_epoch)
		torch.save(varnet.state_dict(), str(save_dir / 'varnet.h5'))


	elif args['mode'] == 'eval':
		varnet.eval()
		varnet.load_state_dict(torch.load(str(save_dir / 'varnet_best.h5')))
		eval_dataloader = DataLoader(dataset,batch_size=args['batch_size'],shuffle=False,pin_memory=True)
		data_dqdm_loop = tqdm.tqdm(eval_dataloader)
		mse_loss = []
		ssim = []
		for batch_idx, batch in enumerate(data_dqdm_loop):
			data_dqdm_loop.set_description("Evaluation")
			batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
			out = varnet.test_step(batch, batch_idx,save_recon_np=True)
			mse_loss.append(out['test_loss'])
			ssim.append(out['ssim'])
		print('MSE Loss', np.mean(mse_loss))
		print('SSIM', np.mean(ssim))
		with open(save_dir / 'eval_opt.txt', 'a') as eval_txt_file:
			eval_txt_file.write(f'\nMSE: {np.mean(mse_loss)}')
			eval_txt_file.write(f'\nSSIM: {np.mean(ssim)}')

	elif args['mode'] == 'eval_image_per_cascade':
		varnet.eval()
		varnet.load_state_dict(torch.load(str(save_dir / 'varnet_best.h5')))
		eval_dataloader = DataLoader(dataset,batch_size=args['batch_size'],shuffle=False,num_workers=8,pin_memory=True)
		data_dqdm_loop = tqdm.tqdm(eval_dataloader)
		for batch_idx, batch in enumerate(data_dqdm_loop):
			data_dqdm_loop.set_description("Evaluation")
			batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
			varnet.eval_images_per_cascade(batch, batch_idx)

if __name__ == '__main__':
	main()
