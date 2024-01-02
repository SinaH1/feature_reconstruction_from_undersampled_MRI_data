import sys
sys.path.insert(0,'../')
import argparse
from UNet.data_utils import *
from torch.utils.data import DataLoader
from UNet.general_utils import *
import tqdm
import os
from torch import nn
from UNet.loss_functions.dice_loss import *
from UNet.unet_model import UNet

parser = argparse.ArgumentParser(description='UNet arguments')

# Data IO
parser.add_argument('--name', type=str, default='images', help='name of the dataset to use')
parser.add_argument('--root_dir',         type=str, default='../dataset/',         help='directory of the data')

# Training and Testing Configuration
parser.add_argument('--mode', type=str, default='train', help='')
parser.add_argument('--net_type', type=str, default='original', help='original, orig+dx, learned_kernels, orig+kernel')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--gpus', type=str, default='0', help='gpu id to use')
parser.add_argument('--save_dir', type=str, default='../results/UNet/test',
					help='directory of the experiment')
parser.add_argument('--save_loss_plot', type=bool, default=True, help='save plot of mean loss per epoch')
parser.add_argument('--save_loss', type=bool, default=False, help='save loss as numpy array')
parser.add_argument('--save_mean_loss', type=bool, default=True, help='save mean loss per epoch as numpy array')
parser.add_argument('--pretrained_model', type=str,   default=None)
parser.add_argument('--check_validation_per_epoch', type=bool, default=True, help='Evaluate validation set per epoch')

def main():
	args = parser.parse_args()
	print_options(parser, args)
	args = vars(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	args['device'] = device

	loss = BinaryDice_and_BCE_loss({},{})

	if args['net_type'] in ['original','learned_kernels']:
		channels = 1
	elif args['net_type'] in ['orig+dx','orig+kernel']:
		channels = 2
	nnunet = UNet(channels,1)
	nnunet.to(device)

	# setting up data loader
	dataset = Torso_Dataset(**args)
	save_dir = Path(args['save_dir'])
	save_dir.mkdir(parents=True, exist_ok=True)

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
				nnunet.load_state_dict(torch.load(str(args['pretrained_model']))['state_dict'])
		nnunet.train()
		torch.set_grad_enabled(True)
		losses_per_epoch = []
		val_losses_per_epoch = []
		optimizer =  torch.optim.Adam(nnunet.parameters(),lr=args['lr'])

		for epoch in range(args['epoch']):
			nnunet.train()
			data_dqdm_loop = tqdm.tqdm(train_dataloader)
			losses = []
			for batch_idx, batch in enumerate(data_dqdm_loop):
				data_dqdm_loop.set_description(f"Epoch {epoch}")
				# train step
				batch = {k: v.to(device=device, non_blocking=True).to(torch.float32) for k, v in batch.items()}

				output = nnunet(batch['img']) # outpu[0] torch.Size([1, 2, 192, 192])
				l = loss(output, batch['segs'])

				# clear gradients
				optimizer.zero_grad()
				# backward
				l.backward()
				# update parameters
				optimizer.step()
				losses.append(l.item())
				if epoch == 0:
					data_dqdm_loop.set_postfix(loss=l.item())
				else:
					data_dqdm_loop.set_postfix(loss=l.item(), epoch_loss=losses_per_epoch[-1])
			if args['check_validation_per_epoch']:
				nnunet.eval()
				val_losses = []
				data_dqdm_loop = tqdm.tqdm(val_dataloader)
				for batch_idx, batch in enumerate(data_dqdm_loop):
					data_dqdm_loop.set_description(f"Validation {epoch}")
					batch = {k: v.to(device=device, non_blocking=True).to(torch.float32) for k, v in batch.items()}
					output = nnunet(batch['img'])
					l = loss(output, batch['segs'])
					val_losses.append(l.item())
					if epoch==0:
						data_dqdm_loop.set_postfix(loss=l.item())
					else:
						data_dqdm_loop.set_postfix(loss=l.item(), epoch_loss=val_losses_per_epoch[-1])
				val_losses_per_epoch.append(sum(val_losses) / len(val_losses))
				if args['save_mean_loss']:
					np.save(str(save_dir / 'val_losses_per_epoch'), val_losses_per_epoch)
				if val_losses_per_epoch[-1] < best_val_loss:
					best_val_loss = val_losses_per_epoch[-1]
					torch.save(nnunet.state_dict(), str(save_dir / 'nnunet_best.h5'))
					with open(save_dir / 'train_opt.txt', 'r+') as train_txt_file:
						lines = train_txt_file.readlines()
						lines[-1] = f'best validation loss: {best_val_loss} at epoch {epoch}'
						train_txt_file.seek(0)
						train_txt_file.writelines(lines)
			losses_per_epoch.append(sum(losses) / len(losses))
			data_dqdm_loop.set_description("Loss per epoch: {}".format(losses_per_epoch[-1]))
			if args['save_loss']:
				if epoch == 0:
					np.save(str(save_dir / 'losses'), losses)
				else:
					np.save(str(save_dir / 'losses'), np.append(np.load(str(save_dir / 'losses')), losses))
			if args['save_mean_loss']:
				np.save(str(save_dir / 'losses_per_epoch'), losses_per_epoch)
			if args['save_loss_plot']:
				if args['check_validation_per_epoch']:
					save_loss_as_plot(str(save_dir / 'losses_per_epoch.png'), losses_per_epoch, val_losses_per_epoch)
				else:
					save_loss_as_plot(str(save_dir / 'losses_per_epoch.png'), losses_per_epoch)
		torch.save(nnunet.state_dict(), str(save_dir / 'nnunet.h5'))

	if args['mode'] == 'eval':
		shuffle = False
		dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=shuffle,pin_memory=True)
		batch_size = args['batch_size']
		dice = []

		nnunet.load_state_dict(torch.load(str(save_dir / 'nnunet_best.h5')))
		nnunet.eval()
		data_dqdm_loop = tqdm.tqdm(dataloader)
		for batch_idx, batch in enumerate(data_dqdm_loop):
			data_dqdm_loop.set_description("Evaluation")
			batch = {k: v.to(device=device, non_blocking=True).to(torch.float32) for k, v in batch.items()}
			output = nnunet(batch['img']) # outpu[0] torch.Size([1, 2, 192, 192])
			binary_output = (torch.sigmoid(output) > 0.5).float()
			for i in range(batch['img'].shape[0]):
				dice.append(dice_coefficient(batch['segs'][i,...].to('cpu'), binary_output[i,...].to('cpu')).item())
				img_save_dir = args['save_dir'] + '/eval_result_segmentation/'#
				Path(img_save_dir).mkdir(parents=True, exist_ok=True)
				img_save_dir += 'eval_seg_{}.png'.format(batch_size*batch_idx + i+1)
				if args['net_type'] in ['original','learned_kernels']:
					save_result_seg_img(img_save_dir, batch['img'][i,...].squeeze(0), batch['segs'][i, ...].squeeze(0), binary_output[i,...].squeeze(0))
		with open(save_dir / 'eval_opt.txt', 'a') as eval_txt_file:
			eval_txt_file.write(f'\nDC: {np.nanmean(dice)}')
		print('DC:', np.nanmean(dice))

def dice_coefficient(target, predict):
	return 2 * torch.sum(torch.mul(predict, target))/(torch.sum(predict + target))

if __name__ == '__main__':
	main()
