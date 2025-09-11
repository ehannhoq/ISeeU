import torch.nn as nn
import torch


class ISeeUModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ISeeUModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        

    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.leaky_relu(out)
        return out


class ISeeU(nn.Module):
    def __init__(self, block:ISeeUModule, num_blocks:list):
        super(ISeeU, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.layer1 = self._make_block(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_block(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_block(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_block(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bbox_head = nn.Linear(512, 4)
        self.conf_head = nn.Linear(512, 1)


    def _make_block(self, block:ISeeUModule, out_channels:int, num_blocks:int, stride:int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels 
        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        bbox = self.bbox_head(out)
        conf = torch.sigmoid(self.conf_head(out))

        return bbox, conf


