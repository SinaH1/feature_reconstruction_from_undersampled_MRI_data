import sys
sys.path.insert(0,'../')
from adjusted_VarNet.models import VariationalNetwork
from adjusted_VarNet.models_learned_kernels import * #VariationalNetwork_learned_kernels
from UNet.general_utils import *
from UNet.unet_model import UNet
from UNet.loss_functions.dice_loss import *
from concatenated_networks import Concatenated_Networks_kernels
import argparse
from data_utils import *
from torch.utils.data import DataLoader
from pathlib import Path
from adjusted_VarNet.misc_utils import print_options, save_loss_as_plot
import tqdm
from general_utils import *
import torch.nn.functional as F
import math


DEFAULT_OPTS_VARNET = dict(features_out=48, num_act_weights=31, num_stages=10, activation='rbf', optimizer='adam',
						   loss_type='complex', init_scale=0.04, sampling_pattern='cartesian',
						   seed=1, lr=1e-4, momentum=0., error_scale=10, loss_weight=1)

parser = argparse.ArgumentParser(description='Variational network arguments')

# Data IO
parser.add_argument('--name', 					type=str, default='simulated_mri', help='name of the dataset to use')
parser.add_argument('--root_dir',         		type=str, default='../dataset/',         help='directory of the data')
parser.add_argument('--net_type', 				type=str, default='original', help='original, learned_kernels')
parser.add_argument('--learn_feature', 			type=str, default='kernel', help='matrix, derivatives, kernel')
# Training and Testing Configuration
parser.add_argument('--mode', 					type=str, default='train', help='train, eval_varnet, eval_seg')
parser.add_argument('--epoch', 					type=int, default=250, help='number of training epoch')
parser.add_argument('--batch_size', 				type=int, default=16, help='batch size')
parser.add_argument('--gpus', 					type=str, default='0', help='gpu id to use')
parser.add_argument('--save_dir', 				type=str, default='../results/concat_networks/_concat_original_pretrained_VarNet_alpha0.1', help='directory of the experiment')
parser.add_argument('--pretrained_varnet', 		type=str, default=None, help='path to pretrained model')
parser.add_argument('--pretrained_nnunet', 		type=str, default=None, help='path to pretrained model')
parser.add_argument('--input', 					type=str, default='cg', help='input for VarNet: ifft, cg(10 iterations), landweber(50 iterations)')
parser.add_argument('--lr', 						type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', 			type=float, default=3e-5, help='weight decay for optimizer')
parser.add_argument('--momentum', 				type=float, default=0., help='momentum for the optimizer')
parser.add_argument('--loss_weight_varnet_unet', 	type=float, default=0.1, help='weight for the loss_ (1-w)*loss_unet + w*loss_varnet')
parser.add_argument('--weight_varnet', 			type=float, default=1000, help='weight alpha (1-w)*loss_unet + w*alpha * loss_varnet')
parser.add_argument('--error_scale', 				type=float, default=1., help='how much to magnify the error map for display purpose')
parser.add_argument('--save_loss_plot', 			type=bool, default=True, help='save plot of mean loss per epoch')
parser.add_argument('--save_mean_loss', 			type=bool, default=True, help='save mean loss per epoch as numpy array')
parser.add_argument('--check_validation_per_epoch', type=bool, default=True, help='Evaluate validation set per epoch')
parser.add_argument('--freeze_unet', 				type=bool, default=False, help='fix weights of unet')
parser.add_argument('--freeze_varnet', 			type=bool, default=False, help='fix weights of varnet')

def main():
	parse_args = parser.parse_args()
	print_options(parser, parse_args)
	parse_args = vars(parse_args)

	args = DEFAULT_OPTS_VARNET
	for key in parse_args.keys():
		args[key] = parse_args[key]

	os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	args['device'] = device
	args['image_size'] = [192,192]
	args['num_coils'] = 4
	args['loss'] = BinaryDice_and_BCE_loss({}, {})

	# setting up network
	if args['net_type'] in ['original','learned_kernels']:
		if args['net_type'] == 'original':
			args['num_input_channels'] = 1
			varnet = VariationalNetwork_learned_kernels(**args).to(device)
			if args['pretrained_varnet']:
				pretrained_model = torch.load(str(args['pretrained_varnet']))
				varnet.load_state_dict(pretrained_model)
		elif args['net_type'] == 'learned_kernels':
			args['num_input_channels'] = 4
			varnet = VariationalNetwork_learned_kernels(**args).to(device)
			if args['pretrained_varnet']:
				pretrained_model = torch.load(str(args['pretrained_varnet']))
				for i in range(10):
					pretrained_model[f'cell_list.{i+1}.conv_kernel'] = pretrained_model[f'cell_list.{i+1}.conv_kernel'].repeat(1,args['num_input_channels'],1,1,1)
				varnet.load_state_dict(pretrained_model, strict=False)

		nnunet = UNet(args['num_input_channels'],1)
		if args['pretrained_nnunet']:
			pretrained_model = torch.load(str(args['pretrained_nnunet']))
			nnunet.load_state_dict(pretrained_model)

		if args['freeze_unet']:
			for param in nnunet.parameters():
				param.requires_grad = False
		if args['freeze_varnet']:
			for i in range(10):
				i+=1
				for param in varnet.cell_list[i].parameters():
					param.requires_grad = False

		concat_net = Concatenated_Networks_kernels(varnet, nnunet).to(device)

		param_group_varnet_cells = []
		for i in range(args['num_stages']):
			param_group_varnet_cells.append({'params': concat_net.varnet.cell_list[i+1].parameters(), 'lr':args['lr'], 'weight_decay':args['weight_decay'], 'momentum':0.99})

		param_group_varnet_cells.append({'params': concat_net.varnet.cell_list[0].parameters(), 'lr':args['lr'], 'weight_decay':args['weight_decay'], 'momentum':0.99})
		param_group_varnet_cells.append({'params': concat_net.segm_unet.parameters(), 'lr':args['lr'], 'weight_decay':args['weight_decay'], 'momentum':0.99})

		optimizer =  torch.optim.Adam(concat_net.parameters(),lr=args['lr'])

	# setting up data loader
	dataset = Mri_Segmentation_Dataset(**args)
	save_dir = Path(args['save_dir'])
	save_dir.mkdir(parents=True, exist_ok=True)

	# start training
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

		concat_net.train()
		torch.set_grad_enabled(True)
		losses_per_epoch = []
		l_seg_per_epoch = []
		l_mse_per_epoch = []
		val_losses_varnet_per_epoch = []
		val_losses_varnet_feat_per_epoch = []
		l_val_losses_per_epoch = []
		l_val_mse_per_epoch = []
		l_val_seg_per_epoch = []

		for epoch in range(args['epoch']):
			concat_net.train()
			data_dqdm_loop = tqdm.tqdm(train_dataloader)
			losses = []
			l_seg = []
			l_mse = []
			best_val_counter = 0
			for batch_idx, batch in enumerate(data_dqdm_loop):
				data_dqdm_loop.set_description(f"Epoch {epoch}")
				# train step
				batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
				output, output_varnet = concat_net(batch)

				l = (1-args['loss_weight_varnet_unet']) * args['loss'](output, batch['seg']) + args['loss_weight_varnet_unet'] * args['weight_varnet'] * F.mse_loss(output_varnet, concat_net.get_varnet_reference(batch['reference']).to(output_varnet.device))
				l_seg.append(args['loss'](output, batch['seg']).item())
				l_mse.append(F.mse_loss(output_varnet, concat_net.get_varnet_reference(batch['reference']).to(output_varnet.device)).item())
				# clear gradients
				optimizer.zero_grad()
				# backward
				l.backward()
				# update parameters
				optimizer.step()

				losses.append(l.item())
				if epoch == 0:
					data_dqdm_loop.set_postfix(loss=l.item(), seg=l_seg[-1], mse = l_mse[-1])
				else:
					data_dqdm_loop.set_postfix(loss=l.item(), epoch_loss=losses_per_epoch[-1], epoch_seg = l_seg_per_epoch[-1],l_mse = l_mse_per_epoch[-1])

			if args['check_validation_per_epoch']:
				concat_net.eval()
				val_losses_varnet = []
				val_losses_varnet_feat = []
				val_losses_seg = []
				l_val_losses = []
				l_val_seg = []
				l_val_mse = []
				data_dqdm_loop = tqdm.tqdm(val_dataloader)

				for batch_idx, batch in enumerate(data_dqdm_loop):
					data_dqdm_loop.set_description(f"Validation {epoch}")
					batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
					output, output_varnet = concat_net(batch)
					l_val = (1-args['loss_weight_varnet_unet']) * args['loss'](output, batch['seg']) + args['loss_weight_varnet_unet'] * args['weight_varnet'] * F.mse_loss(output_varnet, concat_net.get_varnet_reference(batch['reference']).to(output_varnet.device))
					l_val_seg.append(args['loss'](output, batch['seg']).item())
					l_val_mse.append(F.mse_loss(output_varnet, concat_net.get_varnet_reference(batch['reference']).to(output_varnet.device)).item())

					l_val_losses.append(l_val.item())

					if epoch == 0:
						data_dqdm_loop.set_postfix(mse=l_val_mse[-1], dice=l_val_seg[-1], loss = l_val.item())
					else:
						data_dqdm_loop.set_postfix(mse=l_val_mse[-1], dice=l_val_seg[-1], loss = l_val.item(),
											   epoch_mse=l_val_mse_per_epoch[-1], epoch_dice=l_val_seg_per_epoch[-1])
				l_val_losses_per_epoch.append(sum(l_val_losses) / len(l_val_losses))
				l_val_mse_per_epoch.append(sum(l_val_mse) / len(l_val_mse))
				l_val_seg_per_epoch.append(sum(l_val_seg) / len(l_val_seg))
				if args['net_type'] == 'orig+learned_feat' or args['net_type'] == 'orig+dx':
					val_losses_varnet_feat_per_epoch.append(sum(val_losses_varnet_feat)/ len(val_losses_varnet_feat))
				np.save(str(save_dir / 'val_losses_per_epoch'), l_val_losses_per_epoch)
				np.save(str(save_dir / 'val_losses_varnet_per_epoch'), l_val_mse_per_epoch)
				np.save(str(save_dir / 'val_losses_seg_per_epoch'), l_val_seg_per_epoch)
				if l_val_losses_per_epoch[-1] <= best_val_loss:
					if args['net_type'] in ['original','learned_feat','learned_kernels','first_derivative']:
						torch.save(concat_net.varnet.state_dict(), str(save_dir / 'varnet_best.h5'))
					elif args['net_type'] == 'orig+learned_feat' or  args['net_type'] == 'orig+dx':
						torch.save(concat_net.first_varnet.state_dict(), str(save_dir / 'varnet_best.h5'))
						torch.save(concat_net.sec_varnet.state_dict(), str(save_dir / 'varnet_feat_best.h5'))
					torch.save(concat_net.segm_unet.state_dict(), str(save_dir / 'nnunet_best.h5'))
					best_val_loss = l_val_losses_per_epoch[-1]
					with open(save_dir / 'train_opt.txt', 'r+') as train_txt_file:
						lines = train_txt_file.readlines()
						lines[-1] = f'best validation loss: {best_val_loss} at epoch {epoch}'
						train_txt_file.seek(0)
						train_txt_file.writelines(lines)
					best_val_counter = 0
				else:
					best_val_counter += 1
			losses_per_epoch.append(sum(losses) / len(losses))
			l_seg_per_epoch.append(sum(l_seg) / len(l_seg))
			l_mse_per_epoch.append(sum(l_mse) / len(l_mse))
			data_dqdm_loop.set_description("Loss per epoch: {}".format(losses_per_epoch[-1]))
			if args['save_mean_loss']:
				np.save(str(save_dir / 'losses_per_epoch'), losses_per_epoch)
				np.save(str(save_dir / 'losses_seg_per_epoch'), l_seg_per_epoch)
				np.save(str(save_dir / 'losses_mean_per_epoch'), l_mse_per_epoch)
			if args['save_loss_plot']:
				if args['check_validation_per_epoch']:
					save_loss_as_plot(str(save_dir / 'losses_per_epoch.png'), losses_per_epoch, l_val_losses_per_epoch)
					save_loss_as_plot(str(save_dir / 'losses_per_epoch_seg.png'), l_seg_per_epoch, l_val_seg_per_epoch)
					save_loss_as_plot(str(save_dir / 'losses_per_epoch_varnet.png'), l_mse_per_epoch, l_val_mse_per_epoch)
				else:
					save_loss_as_plot(str(save_dir / 'losses_per_epoch.png'), losses_per_epoch)
			if best_val_counter > 30:
				if args['net_type'] in ['original','learned_feat','learned_kernels','first_derivative']:
					torch.save(concat_net.varnet.state_dict(), str(save_dir / f'varnet_at{epoch}.h5'))
				elif args['net_type'] == 'orig+learned_feat' or  args['net_type'] == 'orig+dx':
					torch.save(concat_net.first_varnet.state_dict(), str(save_dir / f'varnet_at{epoch}.h5'))
					torch.save(concat_net.sec_varnet.state_dict(), str(save_dir / f'varnet_feat_at{epoch}.h5'))
				torch.save(concat_net.segm_unet.state_dict(), str(save_dir / f'nnunet_at{epoch}.h5'))
				break

		torch.save(concat_net.varnet.state_dict(), str(save_dir / 'varnet.h5'))
		torch.save(concat_net.segm_unet.state_dict(), str(save_dir / 'nnunet.h5'))

	elif args['mode'] == 'eval_varnet':
		eval_dataloader = DataLoader(dataset,batch_size=args['batch_size'],shuffle=False,num_workers=8,pin_memory=True)
		data_dqdm_loop = tqdm.tqdm(eval_dataloader)
		if args['net_type'] in ['original','learned_feat','learned_kernels','first_derivative']:
			varnet.load_state_dict(torch.load(str(save_dir / 'varnet_best.h5')))
			#print(varnet.cell_list[0].filter_kernels[0])
			#print(varnet.cell_list[0].filter_kernels[1])
			#print(varnet.cell_list[0].filter_kernels[2])
			#print(varnet.cell_list[0].filter_kernels[3])
			mse_loss = []
			ssim = []
			for batch_idx, batch in enumerate(data_dqdm_loop):
				data_dqdm_loop.set_description("Evaluation")
				batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
				out = varnet.test_step(batch, batch_idx)
				mse_loss.append(out['test_loss'])
				ssim.append(out['ssim'])

			print('MSE Loss', np.mean(mse_loss))
			print('MSE Loss std', np.std(mse_loss))
			print('SSIM', np.mean(ssim))
			print('SSIM Std', np.std(ssim))
			np.save(save_dir /'eval_mse.npy', mse_loss)
			np.save(save_dir /'eval_ssim.npy', ssim)
			with open(save_dir / 'eval_varnet_opt.txt', 'a') as eval_txt_file:
				eval_txt_file.write(f'\nMSE: {np.mean(mse_loss)}, \nMSE std: {np.std(mse_loss)}')
				eval_txt_file.write(f'\nSSIM: {np.mean(ssim)}, \nSSIM std: {np.std(ssim)}')


	elif args['mode'] == 'eval_seg':
		eval_dataloader = DataLoader(dataset,batch_size=args['batch_size'],shuffle=False,num_workers=8,pin_memory=True)
		args['batch_size'] = 1
		varnet.load_state_dict(torch.load(str(save_dir / 'varnet_best.h5')))
		nnunet.load_state_dict(torch.load(str(save_dir / 'nnunet_best.h5')))
		concat_net = Concatenated_Networks_kernels(varnet, nnunet).to(device)
		concat_net.eval()
		val_losses_seg = []
		conf_seg = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
		data_dqdm_loop = tqdm.tqdm(eval_dataloader)
		for batch_idx, batch in enumerate(data_dqdm_loop):
			data_dqdm_loop.set_description(f"Segmentation Val ")
			batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
			output, output_varnet = concat_net(batch)

			binary_output = (torch.sigmoid(output) > 0.5).float()
			for i in range(output_varnet.shape[0]):
				TP, TN, FP, FN = get_TP_TN_FP_FN(batch['seg'][i,...].to('cpu'), binary_output[i,...].to('cpu'))
				conf_seg['TP'] += TP/569
				conf_seg['TN'] += TN/569
				conf_seg['FP'] += FP/569
				conf_seg['FN'] += FN/569
				l_seg = dice_coefficient(batch['seg'][i,...].to('cpu'), binary_output[i,...].to('cpu')).item()
				#l_seg = dice_coefficient(batch['seg'][i,...].squeeze(0).squeeze(0), output)
				#print(l_seg)
				val_losses_seg.append(l_seg)
				#l = args['loss'](output, batch['seg'])
				#val_losses.append(l.item())
				#data_dqdm_loop.set_postfix(loss=l.item())
				img_save_dir = args['save_dir'] + '/eval_result_segmentation/'#
				Path(img_save_dir).mkdir(parents=True, exist_ok=True)
				img_save_dir += 'eval_seg_{}.png'.format(args['batch_size']*batch_idx + i)
				save_result_seg_img(img_save_dir, batch['reference'][i, :,:,0], batch['seg'][i, ...].squeeze(0), binary_output[i, ...].squeeze(0))
				#save_result_seg_img(img_save_dir, output_varnet[i,0, ...].detach().squeeze(0), batch['seg'][i, ...].squeeze(0).squeeze(0), output[0])
		np.save(save_dir /'eval_dc.npy', val_losses_seg)
		dc_294 = val_losses_seg[294]
		val_losses_seg = [value for value in val_losses_seg if not math.isnan(value)]  # nan if no liver is in image and prediction

		print('DC:', np.mean(val_losses_seg))
		print('DC std:', np.std(val_losses_seg))
		print('DC 294:', dc_294)
		with open(save_dir / 'eval_seg_opt.txt', 'a') as eval_txt_file:
			eval_txt_file.write('\nDC: {}\nDC std: {}\nDC_test_294: {} \n\nTP: {} \nTN:{} \nFP: {} \nFN: {}\nPrecision: {}\nRecall: {}\n'.format(np.mean(val_losses_seg), np.std(val_losses_seg), dc_294 ,conf_seg['TP'], conf_seg['TN'], conf_seg['FP'], conf_seg['FN'], conf_seg['TP']/(conf_seg['TP']+conf_seg['FP']), conf_seg['TP']/(conf_seg['TP']+conf_seg['FN'])))
		#plt.hist(val_losses_seg, bins=100, density=True, alpha=0.75, color='b', edgecolor='black')
		#plt.title('Histogramm')
		#plt.xlabel('Werte')
		#plt.ylabel('Häufigkeit')
		#plt.savefig(save_dir / 'DC_histogramm.png')


if __name__ == '__main__':
	main()
