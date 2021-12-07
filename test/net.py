import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return x

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = FFTConv1d(nn.Conv2d(1, 32, 3, 1))
        self.conv2 = FFTConv1d(nn.Conv2d(32, 64, 3, 1))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x)
        x = F.relu(x)   
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return x

class FFTConv1d(nn.Module):
    def __init__(self, conv):
        super(FFTConv1d, self).__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.padding = conv.padding

    def fftConv1D(self, x, W):
        channel, x_row, x_col = x.size()
        x = torch.reshape(x, (channel, x_row*x_col))

        channel, W_row, W_col = W.size()
        W_pad = nn.ZeroPad2d((0, x_col-W_col, 0, x_row-W_row))(W)
        W_pad = torch.reshape(W_pad, (channel, W_pad.size(dim=1)*W_pad.size(dim=2)))

        x_fft = torch.fft.fft(x)
        W_fft = torch.fft.fft(torch.flipud(torch.flip(W_pad, [0,1])))
        y = (torch.fft.ifft(x_fft * W_fft)).real

        ans_row, ans_col = x_row - W_row + 1, x_col - W_col + 1
        y = torch.roll(y, 1)
        y = torch.reshape(y, (channel, x_row, x_col))
        y = torch.sum(y, dim=0)
        y = y[:ans_row, :ans_col]

        return y

    def fftConv1D_channel(self, x, W, b):
        total_size, in_channel, x_row, x_col = x.size()
        out_channel, in_channel, W_row, W_col = W.size()
        ans_row, ans_col = (x_row - W_row + 1), (x_col - W_col + 1)
        ans = torch.zeros(total_size, out_channel, ans_row, ans_col)

        for img in range(total_size):
            for out_ch in range(out_channel):
                ans[img,out_ch] = self.fftConv1D(x[img], W[out_ch]) + b[out_ch]

        return ans

    def forward(self, x):
        x = nn.ZeroPad2d(self.padding[0])(x)
        ans = self.fftConv1D_channel(x, self.weight, self.bias)
        return ans
