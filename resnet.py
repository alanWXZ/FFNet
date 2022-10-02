import sys
from collections import OrderedDict
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import functools
from fre_matching_new import *



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, BatchNorm2d=nn.BatchNorm2d, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
        self.inplanes = 128
        self.depth_inplanes=128
        self.is_fpn = is_fpn
        super(ResNet, self).__init__()
        self.business_layer = []
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if dilation[1]!=1 else 2, dilation=dilation[1], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1 if dilation[2]!=1 else 2, dilation=dilation[2], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if dilation[3]!=1 else 2, dilation=dilation[3], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.DCT_spatial_attention_decomposition = DCT_spatial_attention_decomposition()
        self.spatialSELayer=ChannelSELayer(128)
        self.spatialSELayer1 = ChannelSELayer(256)
        self.conv_f_1=nn.Conv2d(4096,2048,1,stride=1)
        self.conv_f_2 = nn.Conv2d(2048, 2048, 1, stride=1)
        self.business_layer.append(self.conv_f_1)
        self.business_layer.append(self.conv_f_2)
        self.business_layer.append(self.spatialSELayer)
        self.business_layer.append(self.spatialSELayer1)
        self.depth_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)
        )
        self.dlayer1 = self._make_depth_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.dlayer2 = self._make_depth_layer(block, 128, layers[1], stride=1 if dilation[1]!=1 else 2, dilation=dilation[1], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.dlayer3 =self._make_depth_layer(block, 256, layers[2], stride=1 if dilation[2]!=1 else 2, dilation=dilation[2], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.dlayer4 =self._make_depth_layer(block, 512, layers[3], stride=1 if dilation[3]!=1 else 2, dilation=dilation[3], bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d)
        self.business_layer.append(self.depth_layer)
        self.business_layer.append(self.dlayer1)
        self.business_layer.append(self.dlayer2)
        self.business_layer.append(self.dlayer3)
        self.business_layer.append(self.dlayer4)

        self.conv_depth_hha=nn.Conv2d(1, 3, 3, stride=1, padding=1, bias=False)
        self.business_layer.append(self.conv_depth_hha)

        self.business_layer+=self.DCT_spatial_attention_decomposition.business_layer
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003, BatchNorm2d=nn.BatchNorm2d, se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = True, momentum=bn_momentum))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d,se=se))
            aaa=block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d,se=se)

            self.business_layer.append(aaa.se)
        return nn.Sequential(*layers)
    def _make_depth_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003, BatchNorm2d=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = True, momentum=bn_momentum))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.depth_inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), bn_momentum=bn_momentum, BatchNorm2d=BatchNorm2d))

        return nn.Sequential(*layers)
    def forward(self, x_in,y_in, start_module=1, end_module=5):
        if start_module <= 1:
            x=self.conv1(x_in)
            x=self.bn1(x)
            x=self.relu1(x)

            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)

            y_in=y_in.unsqueeze(0)
            y_in = y_in.unsqueeze(0)
            y_in = F.interpolate(y_in, scale_factor=2, mode='bilinear', align_corners=True)
            y_in=self.conv_depth_hha(y_in)
            y=self.depth_layer(y_in)

            x, y = self.DCT_spatial_attention_decomposition(x, y, x_in, y_in)

            y=self.dlayer1(y)
            y = self.dlayer2(y)
            y = self.dlayer3(y)
            y = self.dlayer4(y)
            start_module = 2
        features = []
        for i in range(start_module, end_module+1):

            x=eval('self.layer%d'%(i-1))(x)

            features.append(x)
        x_y = x+y
        if self.is_fpn:
            if len(features) == 1:
                return features[0]
            else:
                return tuple(features)
        else:
            return x_y


def get_resnet101(num_classes=19, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False, BatchNorm2d=nn.BatchNorm2d):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn, BatchNorm2d=BatchNorm2d)
    return model

def get_resnet50(num_classes=19, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False, BatchNorm2d=nn.BatchNorm2d):
    model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes, dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn, BatchNorm2d=BatchNorm2d)
    return model



if __name__ == '__main__':
    net = get_resnet50().cuda()