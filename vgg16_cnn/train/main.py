import torch
import random
import numpy as np

from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.models as models

from vgg16 import VGG16

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


device = 'cuda' if torch.cuda.is_available() else 'cpu' # device = 'cpu' # CPU ONLY


net = VGG16(num_classes=10)
if torch.cuda.is_available():
	net = net.cuda()

if device == 'cuda':
	# make it concurent
	# net = torch.nn.DataParallel(net)
	cudnn.benchmark = True


def train_dataset(path, shuffle=True):

	transformation = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 244)),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor()
		])

	dataset = torchvision.datasets.ImageFolder(path, transform=transformation)
	loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0, shuffle=shuffle)
	return loader

def val_dataset(path, shuffle=False):

	transformation = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 244)),
		torchvision.transforms.ToTensor()
		])

	dataset = torchvision.datasets.ImageFolder(path, transform=transformation)
	loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0, shuffle=shuffle)
	return loader

# https://github.com/fastai/imagenette
# https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz

train_loader = train_dataset('../../data/imagenette2/train', shuffle=True)
val_loader = val_dataset('../../data/imagenette2/val', shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


def train(epoch):
	net.train()
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0

	with tqdm(total=len(train_loader), desc='Train', leave=True) as progress_bar:
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()

			outputs = net(inputs)

			loss = criterion(outputs, targets)
			loss.backward()

			optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			acc = correct / total
			progress_bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
					% (train_loss / (batch_idx + 1), 100. * acc, correct, total))
			progress_bar.update()
		scheduler.step(acc)

	torch.save(net.state_dict(), "vgg16_cnn.pt")


def test(epoch):
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


def run(epochs):
	for epoch in range(epochs):
		train(epoch)
		test(epoch)


if __name__ == '__main__':
	print(device)
	run(100)
