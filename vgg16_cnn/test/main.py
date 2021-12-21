import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

from net import *

def make_layers_fft(cfg, thresh=0, batch_norm=True):
	layers = []
	in_channels = 3
	for k in cfg:
		if k == 'M':
			layers.extend([nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
		else:
			conv2d = nn.Conv2d(in_channels=in_channels,
					out_channels=k,
					kernel_size=3,
					padding=1)
			print(f'[thresh] {thresh}')
			conv2d = FFTConv1d_thresh(conv2d, thresh)

			if batch_norm:
				layers.extend([conv2d, nn.BatchNorm2d(num_features=k), nn.ReLU(inplace=True)])
			else:
				layers.extend([conv2d, nn.ReLU(inplace=True)])
			in_channels = k

	return nn.Sequential(*layers)

def make_layers(cfg, batch_norm=True):
	layers = []
	in_channels = 3
	for k in cfg:
		if k == 'M':
			layers.extend([nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
		else:
			conv2d = nn.Conv2d(in_channels=in_channels,
					out_channels=k,
					kernel_size=3,
					padding=1)
			if batch_norm:
				layers.extend([conv2d, nn.BatchNorm2d(num_features=k), nn.ReLU(inplace=True)])
			else:
				layers.extend([conv2d, nn.ReLU(inplace=True)])
			in_channels = k

	return nn.Sequential(*layers)


cfgs = {
		'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		'B': [],
		'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
		'D': [],
		'E': []
		}

def VGG16(num_classes, pretrained=False):
	features = make_layers(cfg=cfgs['C'])
	vgg = VGG(features, num_classes=num_classes)
	return vgg

def VGG16_fft(num_classes, pretrained=False):
	features = make_layers_fft(cfg=cfgs['C'], thresh=1)
	vgg = VGG(features, num_classes=num_classes)
	return vgg

def val_dataset(path, shuffle=False):

	transformation = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 244)),
		torchvision.transforms.ToTensor()
		])

	dataset = torchvision.datasets.ImageFolder(path, transform=transformation)
	loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0, shuffle=shuffle)
	return loader

def test(epoch, net):
	net.eval()
	train_loss = 0
	correct = 0
	total = 0

	with tqdm(total=len(val_loader), desc='Val', leave=True) as progress_bar:
		for batch_idx, (inputs, targets) in enumerate(val_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)

			loss = criterion(outputs, targets)

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			acc = correct / total
			progress_bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
					% (train_loss / (batch_idx + 1), 100. * acc, correct, total))
			progress_bar.update()

if __name__ == "__main__":
	device = "cpu"
	print(f'[device] {device}')

	path = '../train/vgg16_cnn.pt'

	net = VGG16(num_classes=10)
	vgg = net.to(device)
	vgg.load_state_dict(torch.load(path))

	net_fft = VGG16_fft(num_classes=10)
	vgg_fft = net_fft.to(device)
	vgg_fft.load_state_dict(torch.load(path))

	val_loader = val_dataset('../../data/imagenette2/val', shuffle=False)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

	test(1, vgg_fft)
