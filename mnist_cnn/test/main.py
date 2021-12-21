import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from net import *

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	count = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			count += 1
			print(f'NO. {count} acc: {100. * correct / (count * 100)}')

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
	device = torch.device("cpu")

	test_kwargs = {'batch_size': 100}
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])
	dataset = datasets.MNIST('../../data', train=False, transform=transform)
	test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

	trained_path = "../train/mnist_cnn.pt"
	model = Net().to(device)
	#model = CustomNet().to(device)
	model.load_state_dict(torch.load(trained_path))

	test(model, device, test_loader)
