import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

class FFTConv1d_thresh(nn.Module):
	def __init__(self, conv, thresh=0):
		super(FFTConv1d_thresh, self).__init__()
		self.weight = conv.weight
		self.bias = conv.bias
		self.padding = conv.padding
		self.stride = conv.stride
		self.thresh = thresh

	def get_xfft_thresh(self, x):
		total, channel, x_row, x_col = x.size()
		x = x.view(total, channel, x_row*x_col)
		x_fft = torch.fft.fft(x)

		zero_idx = torch.nonzero((abs(x_fft.real) < self.thresh), as_tuple=True)
		x_fft[zero_idx] = 0

		return x_fft

	def get_Wfft(self, x, W):
		total, channel, x_row, x_col = x.size()
		out_channel, channel, W_row, W_col = W.size()

		W_pad = nn.ZeroPad2d((0, x_col-W_col, 0, x_row-W_row))(W)
		W_pad_re = W_pad.view(out_channel, channel, W_pad.size(dim=2)*W_pad.size(dim=3))
		W_fft = torch.fft.fft(
				torch.flip(
					torch.flip(W_pad_re, [0,1]), [0,1,2])
				)

		return W_fft

	def fftConv1D(self, x_fft, x_row, x_col, W_fft, W_row, W_col, channel):
		y = torch.fft.ifft(x_fft * W_fft).real

		ans_row, ans_col = ceil((x_row - W_row + 1) / self.stride[0]), \
						ceil((x_col - W_col + 1) / self.stride[1])

		y = torch.roll(y, 1)
		y = y.view(channel, x_row, x_col)
		y = torch.sum(y, dim=0)
		y = y[::self.stride[0], ::self.stride[1]]
		y = y[:ans_row, :ans_col]

		return y

	def fftConv1D_channel(self, x, W, b):
		total, in_channel, x_row, x_col = x.size()
		out_channel, in_channel, W_row, W_col = W.size()
		ans_row, ans_col = ceil((x_row - W_row + 1) / self.stride[0]), \
						ceil((x_col - W_col + 1) / self.stride[1])
		ans = torch.zeros(total, out_channel, ans_row, ans_col)

		x_fft = self.get_xfft_thresh(x)
		W_fft = self.get_Wfft(x, W)
		tuple_total, tuple_out = tuple(range(total)), tuple(range(out_channel))
		for img, out_ch in itertools.product(tuple_total, tuple_out):
			ans[img,out_ch] = self.fftConv1D(x_fft[img], x_row, x_col,
											W_fft[out_ch], W_row, W_col,
											in_channel) + b[out_ch]

		return ans

	def forward(self, x):
		with torch.no_grad():
			x = nn.ZeroPad2d((self.padding[0],self.padding[0],
				self.padding[1],self.padding[1]))(x)
			x = self.fftConv1D_channel(x, self.weight, self.bias)
		print(f'[result] {x.size()}')

		return x
