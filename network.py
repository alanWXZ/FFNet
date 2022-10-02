# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from config import config
from resnet import get_resnet50
from aic_module import Bottleneck3D


class ASPP(nn.Module):
    def __init__(self, in_channel=128, depth=128):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool3d((1,1,1))  # (1,1)means ouput_dim
        self.conv = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv3d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv3d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv3d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv3d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out


'''
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
'''

'''
Input: 60*36*60 sketch
Latent code: 15*9*15
'''
class CVAE(nn.Module):
    def __init__(self, norm_layer, bn_momentum, latent_size=16):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.mean = nn.Conv3d(self.latent_size, self.latent_size, kernel_size=1, bias=True)      # predict mean.
        self.log_var = nn.Conv3d(self.latent_size, self.latent_size, kernel_size=1, bias=True)     # predict mean.


        self.decoder_x = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.latent_size*2, self.latent_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(self.latent_size, self.latent_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Dropout3d(0.1),
            nn.Conv3d(self.latent_size, 2, kernel_size=1, bias=True)
        )

    def forward(self, x, gt=None):
        b, c, h, w, l = x.shape

        if self.training:
            gt = gt.view(b, 1, h, w, l).float()
            for_encoder = torch.cat([x, gt], dim=1)
            enc = self.encoder(for_encoder)
            pred_mean = self.mean(enc)
            pred_log_var = self.log_var(enc)

            decoder_x = self.decoder_x(x)

            out_samples = []
            out_samples_gsnn = []
            for i in range(config.samples):
                std = pred_log_var.mul(0.5).exp_()
                eps = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()

                z1 = eps * std + pred_mean
                z2 = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()

                sketch = self.decoder(torch.cat([decoder_x, z1], dim=1))
                out_samples.append(sketch)

                sketch_gsnn = self.decoder(torch.cat([decoder_x, z2], dim=1))
                out_samples_gsnn.append(sketch_gsnn)

            sketch = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
            sketch = torch.mean(sketch, dim=0)
            sketch_gsnn = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples_gsnn])
            sketch_gsnn = torch.mean(sketch_gsnn, dim=0)

            return pred_mean, pred_log_var, sketch_gsnn, sketch
        else:
            out_samples = []
            for i in range(config.samples):
                z = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()
                decoder_x = self.decoder_x(x)
                out = self.decoder(torch.cat([decoder_x, z], dim=1))
                out_samples.append(out)
            sketch_gsnn = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
            sketch_gsnn = torch.mean(sketch_gsnn, dim=0)
            return None, None, sketch_gsnn, None

class STAGE1(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE1, self).__init__()
        self.business_layer = []
        norm_layer=nn.BatchNorm3d
        self.conv=nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False)
        self.oper1 = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            #nn.BatchNorm3d(3,0.1),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            #nn.BatchNorm3d(64, 0.1),
            nn.ReLU(),
            nn.Conv3d(64, 512, kernel_size=3, padding=1, bias=False),
            norm_layer(512, momentum=bn_momentum),
            #nn.BatchNorm3d(512, 0.1),
            nn.ReLU(),
        )
        self.oper1_1=nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            #nn.BatchNorm3d(3, 0.1),
            nn.ReLU(),
        )
        self.oper1_2=nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            # norm_layer(64, momentum=bn_momentum),
            nn.BatchNorm3d(64, 0.1),
            nn.ReLU(),
        )
        self.oper1_3=nn.Sequential(
            nn.Conv3d(64, 512, kernel_size=3, padding=1, bias=False),
            # norm_layer(feature, momentum=bn_momentum),
            nn.BatchNorm3d(512, 0.1),
            nn.ReLU(),
        )
        self.oper1_4=nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False)
        self.oper1_5 = nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False)
        self.business_layer.append(self.oper1)

        self.completion_layer1 = nn.Sequential(
            Bottleneck3D(512, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(512, 128,
                          kernel_size=1, stride=1, bias=False),
                #norm_layer(feature, momentum=bn_momentum),
                nn.BatchNorm3d(128, 0.1),
                # nn.ReLU(),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(128, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(128, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.completion_layer1)

        self.completion_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
                # nn.ReLU(),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.completion_layer2)

        self.cvae = CVAE(norm_layer=norm_layer, bn_momentum=bn_momentum, latent_size=config.lantent_size)
        self.business_layer.append(self.cvae)

        self.classify_sketch = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, 2, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_sketch)

    def forward(self, tsdf, depth_mapping_3d, sketch_gt=None):

        tsdf = self.oper1(tsdf)

        completion1 = self.completion_layer1(tsdf)
        completion2 = self.completion_layer2(completion1)

        up_sketch1 = self.classify_sketch[0](completion2)
        up_sketch1 = up_sketch1 + completion1
        up_sketch2 = self.classify_sketch[1](up_sketch1)
        pred_sketch_raw = self.classify_sketch[2](up_sketch2)

        _, pred_sketch_binary = torch.max(pred_sketch_raw, dim=1, keepdim=True)        # (b, 1, 60, 36, 60) binary-voxel sketch
        pred_mean, pred_log_var, pred_sketch_gsnn, pred_sketch= self.cvae(pred_sketch_binary.float(), sketch_gt)

        return pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var


class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE2, self).__init__()
        self.business_layer = []
        norm_layer=nn.BatchNorm2d
        norm1_layer=nn.BatchNorm3d
        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        self.business_layer.append(self.downsample)

        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit

        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)

        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm1_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )

        self.business_layer.append(self.semantic_layer1)
        self.semantic_layer0 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer0)
        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm1_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)

        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm1_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm1_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_semantic)

        self.oper_sketch = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm1_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm1_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm1_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.oper_sketch_cvae = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm1_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm1_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm1_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.gt_2d=nn.Sequential(nn.Conv2d(1, 12, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(12, momentum=bn_momentum),
                                 nn.ReLU(inplace=False),
                                 nn.Conv2d(12, 32, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(32, momentum=bn_momentum),
                                 nn.ReLU(inplace=False),
                                 nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(64, momentum=bn_momentum),
                                 nn.ReLU(inplace=False),
                                 nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(32, momentum=bn_momentum),
                                 nn.ReLU(inplace=False),
                                 nn.Conv2d(32, 12, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(12, momentum=bn_momentum),
                                 nn.ReLU(inplace=False),
        )
        self.gt_2d_c128 = nn.Sequential(nn.Conv2d(12, 32, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(32, momentum=bn_momentum),
                                   nn.ReLU(inplace=False),
                                   nn.Conv2d(32, 128, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(128, momentum=bn_momentum),
                                   nn.ReLU(inplace=False),
                                   )

        self.gt_2d_c128_direct = nn.Sequential(nn.Conv2d(1, 12, 3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(12, momentum=bn_momentum),
                                        nn.ReLU(inplace=False),
                                        nn.Conv2d(12, 32, 3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(32, momentum=bn_momentum),
                                        nn.ReLU(inplace=False),
                                        nn.Conv2d(32, 128, 3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(128, momentum=bn_momentum),
                                        nn.ReLU(inplace=False),
                                        )
        self.business_layer.append(self.oper_sketch)
        self.business_layer.append(self.oper_sketch_cvae)
        self.aspp=ASPP()
        self.business_layer.append(self.gt_2d)
        self.gt_c=nn.Conv3d(140, 128, 3, stride=1, padding=1, bias=False)
        self.business_layer.append(self.gt_c)
        self.business_layer.append(self.gt_2d_c128)
        self.business_layer.append(self.gt_2d_c128_direct)

    def forward(self, feature2d, depth_mapping_3d, pred_sketch_raw, pred_sketch_gsnn,gt_2d):
        # reduce the channel of 2D feature map
        if self.resnet_out != self.feature:
            feature2d = self.downsample(feature2d)
        feature2d = F.interpolate(feature2d, scale_factor=8, mode='bilinear', align_corners=True)

        gt_2d = gt_2d.view(1, 1, 480, 640)
        gt_2d = F.interpolate(gt_2d, scale_factor=0.5, mode='bilinear', align_corners=True)
        gt_2d = self.gt_2d(gt_2d)

        b1, c1, h1, w1 = gt_2d.shape
        gt_2d = gt_2d.view(b1, c1, h1 * w1).permute(0, 2, 1)

        b, c, h, w = feature2d.shape
        feature2d = feature2d.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

        zerosVec = torch.zeros(b, 1,c).cuda()
        segVec = torch.cat((feature2d, zerosVec), 1)

        segres = [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(b)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)  # B, (channel), 60, 36, 60

        zerosVec_1 = torch.zeros(b, 1, c1).cuda()
        segVec_1 = torch.cat((gt_2d, zerosVec_1), 1)

        segres_1 = [torch.index_select(segVec_1[i], 0, depth_mapping_3d[i]) for i in range(b)]
        segres_1 = torch.stack(segres_1).permute(0, 2, 1).contiguous().view(b, c1, 60, 36, 60)

        if self.ThreeDinit:
            pool = self.pooling(segres)

            zero = (segres == 0).float()
            pool = pool * zero
            segres = segres + pool

        if self.ThreeDinit:
            pool1 = self.pooling(segres_1)

            zero1 = (segres_1 == 0).float()
            pool1 = pool1 * zero1
            segres_1 = segres_1 + pool1


        sketch_proi = self.oper_sketch(pred_sketch_raw)
        sketch_proi_gsnn = self.oper_sketch_cvae(pred_sketch_gsnn)
        segres = torch.cat((segres, segres_1), 1)

        segres = self.gt_c(segres)

        seg_fea = segres + sketch_proi + sketch_proi_gsnn

        semantic1 = self.semantic_layer1(seg_fea)
        semantic2 = self.semantic_layer2(semantic1)
        up_sem1 = self.classify_semantic[0](semantic2)
        up_sem1 = up_sem1 + semantic1
        up_sem2 = self.classify_semantic[1](up_sem1)
        pred_semantic = self.classify_semantic[2](up_sem2)


        return pred_semantic, None


'''
main network
'''
class Network(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Network, self).__init__()
        self.business_layer = []

        if eval:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        else:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        self.business_layer += self.backbone.business_layer
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.stage1 = STAGE1(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage1.business_layer

        self.stage2 = STAGE2(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage2.business_layer
        self.conv=nn.Conv2d(4096,2048,kernel_size=3,padding=1)

    def forward(self, rgb, hha,depth_mapping_3d, tsdf, sketch_gt,gt_2d,depth):



        feature2d= self.backbone(rgb,depth)

        pred_sketch_raw = torch.zeros(4, 2, 60, 36, 60).cuda()
        pred_sketch_gsnn = torch.zeros(4, 2, 60, 36, 60).cuda()
        pred_sketch = torch.zeros(4, 2, 60, 36, 60).cuda()
        pred_mean = torch.zeros(4, 32, 15, 9, 15).cuda()
        pred_log_var = torch.zeros(4, 32, 15, 9, 15).cuda()

        pred_semantic, _ = self.stage2(feature2d, depth_mapping_3d, pred_sketch_raw,
                                                                pred_sketch_gsnn,gt_2d)

        if self.training:
            return pred_semantic, _, pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var
        return pred_semantic, _, pred_sketch_gsnn

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)