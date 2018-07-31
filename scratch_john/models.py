import math
import torch
import torch.nn as nn
from fastai.layers import Flatten
from helpers import Lambda


# Adapted from Travis which was adapted from fastai

# TODO: uncomment the dropout() lines if model is training and overfitting

def conv(ni, nf, ks=3, stride=1):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)

def bn1(planes):
    m = nn.BatchNorm1d(planes)
    m.weight.data.fill_(1)
    m.bias.data.zero_()
    return m

def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

class fc1(nn.Module):
    def __init__(self, ni, nf, ks=2, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(2, stride=2, padding=1)
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.max(out)
        return out
    
class fc2(nn.Module):
    def __init__(self, ni, nf, ks=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        return self.relu(out)
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0):
        super().__init__()
        self.conv1 = conv(inplanes, planes, stride=stride)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = residual + out
        out = self.relu(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return out
    

class AudioResNet(nn.Module):
    def __init__(self, block, layers, num_classes, fully_conv=False, dropout=0):
        super().__init__()
        self.inplanes = 64 # ? 
        self.dropout = dropout
        print("Model head is fully conv?", fully_conv)
        
        features = [
            Lambda(lambda x: x.view(x.shape[0], 1, x.shape[1], x.shape[2])),
            conv(1, 64, ks=3, stride=2),
            bn(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            self.make_layer(block, 64, layers[0], dropout=dropout),
            self.make_layer(block, 128, layers[1], stride=2, dropout=dropout),
            self.make_layer(block, 256, layers[2], stride=2, dropout=dropout),
            self.make_layer(block, 512, layers[3], stride=2, dropout=dropout),
        ]
        out_sz = 512 * block.expansion
        
        if fully_conv:
            features += [
                nn.Conv2d(out_sz, num_classes, 3, padding=1),
                Lambda(lambda x: x.view(x.shape[0], num_classes, -1)),
                Lambda(lambda x: torch.mean(x, dim=2))
            ]
        else:
            features += [
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                #nn.Dropout(0.1),
                nn.Linear(out_sz, num_classes)
            ]
        
        self.features = nn.Sequential(*features)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
        
    def make_layer(self, block, planes, blocks, stride=1, dropout=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes*block.expansion, ks=1, stride=stride),
                bn(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout=dropout))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.features(x)