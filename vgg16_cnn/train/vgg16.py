import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super().__init__()
        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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

if __name__ == '__main__':
    make_layers(cfgs['C'])