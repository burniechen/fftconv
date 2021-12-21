import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

class FFTConv1d_thresh(nn.Module):
	def __init__(self, conv, thresh):
		super(FFTConv1d_thresh, self).__init__()
		self.weight = conv.weight
		self.bias = conv.bias
		self.padding = conv.padding
		self.thresh = thresh

	def get_xfft_thresh(self, x):
		total, channel, x_row, x_col = x.size()
		x = x.view(total, channel, x_row*x_col)
		x_fft = torch.fft.fft(x)

		x_fft = torch.fft.fft(x)
		zero_idx = torch.nonzero((abs(x_fft.real) < self.thresh), as_tuple=True)
		x_fft[zero_idx] = 0;

		return x_fft

	def get_Wfft(self, x, W):
		total, channel, x_row, x_col = x.size()
		out_channel, channel, W_row, W_col = W.size()

		W_pad = nn.ZeroPad2d((0, x_col-W_col, 0, x_row-W_row))(W)
		W_pad_re = W_pad.view(out_channel, channel, W_pad.size(dim=2)*W_pad.size(dim=3))
		W_fft = torch.empty(
				(out_channel, channel, W_pad.size(dim=2)*W_pad.size(dim=3)),
				dtype=torch.complex64
		)
		for i in range(out_channel):
			W_fft[i] = torch.fft.fft(torch.flipud(torch.flip(W_pad_re[i], [0,1])))

		return W_fft

	def fftConv1D(self, x_fft, x_row, x_col, W_fft, W_row, W_col, channel):
		y = (torch.fft.ifft(x_fft * W_fft)).real

		ans_row, ans_col = (x_row - W_row + 1), (x_col - W_col + 1)
		y = torch.roll(y, 1)
		y = y.view(channel, x_row, x_col)
		y = torch.sum(y, dim=0)
		y = y[:ans_row, :ans_col]

		return y

	def fftConv1D_channel(self, x, W, b):
		total, in_channel, x_row, x_col = x.size()
		out_channel, in_channel, W_row, W_col = W.size()
		ans_row, ans_col = (x_row - W_row + 1), (x_col - W_col + 1)
		ans = torch.zeros(total, out_channel, ans_row, ans_col)

		x_fft = self.get_xfft_thresh(x)
		W_fft = self.get_Wfft(x, W)

		for img in range(total):
			for out in range(out_channel):
				ans[img,out] = self.fftConv1D(
						x_fft[img], x_row, x_col,
						W_fft[out], W_row, W_col,
						in_channel) + b[out]

		return ans

	def forward(self, x):
		with torch.no_grad():
			x = nn.ZeroPad2d(self.padding[0])(x)
			x = self.fftConv1D_channel(x, self.weight, self.bias)
		print(f'[fft] {x.size()}')
		return x

class VGG(nn.Module):
	def __init__(self, features, num_classes=1000):
		super().__init__()
		self.features = features
		self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
				nn.Linear(in_features=512 * 7 * 7, out_features=4096),
				nn.ReLU(inplace=True),
				nn.Dropout(.5),
				nn.Linear(in_features=4096, out_features=4096),
				nn.ReLU(inplace=True),
				nn.Dropout(.5),
				nn.Linear(in_features=4096, out_features=num_classes)
				)
		self.sm = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.features(x)
		x = self.avg_pool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

	def inference(self, x):
		x = self.forward(x)
		x = self.sm(x)
		return x
