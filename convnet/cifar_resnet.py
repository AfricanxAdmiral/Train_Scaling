''' Incremental-Classifier Learning
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# ss
from convnet import conv2d_fw


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False, mtl=True, ft=True):
        super(ResNetBasicblock, self).__init__()
        ##########for SS########################################
        if mtl is True and ft is True:
            self.Conv2d = conv2d_fw.Conv2d_fw
            self.BatchNorm2d = conv2d_fw.BatchNorm2d_fw
            self.FeatureWise = conv2d_fw.FeatureWiseTransformation2d_fw
        else:
            self.Conv2d = nn.Conv2d
            self.BatchNorm2d = nn.BatchNorm2d
            self.FeatureWise = nn.BatchNorm2d
        ############################################################
        
        
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = self.FeatureWise(planes)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.featureSize = 64
        self.last = last
        
    def forward(self, x, gamma_beta=None):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = self.relu(basicblock)

        basicblock = self.conv_b(basicblock)
        if gamma_beta is not None:  #Scaling / Shifting in the last BatchNorm layer
            gamma = gamma_beta[:,:64]
            beta = gamma_beta[:,-64:]
            gamma = gamma.view(-1, 64, 1, 1).expand_as(basicblock)
            beta = beta.view(-1, 64, 1, 1).expand_as(basicblock)
            basicblock = gamma*basicblock + beta

        else :
            basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        basicblock +=residual
        
        if not self.last:  #remove ReLU in the last layer
            basicblock = self.relu(basicblock)
        return basicblock


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, zero_init_residual=False, nf=64, mtl=True):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()
        if mtl is True:
            self.Conv2d = conv2d_fw.Conv2d_fw
            self.BatchNorm2d = conv2d_fw.BatchNorm2d_fw
        else:
            self.Conv2d = nn.Conv2d
            self.BatchNorm2d = nn.BatchNorm2d

        self.featureSize = nf
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        #self.num_classes = num_classes
        channels = 3
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1, FT=False)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, FT=False)
        #self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, last_phase=True, FT=True)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
                if m.bias is not None:
                    m.bias.data.zero_()
#                     m.mtl_bias.data.zero_()  

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False, FT=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
            #downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ft=FT))
        self.inplanes = planes * block.expansion
        
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes, ft=FT))
            layers.append(block(self.inplanes, planes, last=True, ft=True))
        else: 
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, ft=FT))

        return nn.Sequential(*layers)
        
    # get the data label and pass them to the corresponding layers          
    def change_fw_label(self, label):             
            for target, param in self.stage_1.named_parameters(): 
                name_t = target.strip()                
                if (name_t == '0.bn_b.label'):
                    param.data = label
                elif (name_t == '1.bn_b.label'):
                    param.data = label
                elif (name_t == '2.bn_b.label'):
                    param.data = label
                elif (name_t == '3.bn_b.label'):
                    param.data = label
                elif (name_t == '4.bn_b.label'):
                    param.data = label
                    
            for target, param in self.stage_2.named_parameters():                
                name_t = target.strip()                
                if (name_t == '0.bn_b.label'):
                    param.data = label
                elif (name_t == '1.bn_b.label'):
                    param.data = label
                elif (name_t == '2.bn_b.label'):
                    param.data = label
                elif (name_t == '3.bn_b.label'):
                    param.data = label
                elif (name_t == '4.bn_b.label'):
                    param.data = label  
                    
            for target, param in self.stage_3.named_parameters():                
                name_t = target.strip()                
                if (name_t == '0.bn_b.label'):
                    param.data = label
                elif (name_t == '1.bn_b.label'):
                    param.data = label
                elif (name_t == '2.bn_b.label'):
                    param.data = label
                elif (name_t == '3.bn_b.label'):
                    param.data = label
                elif (name_t == '4.bn_b.label'):
                    param.data = label
    
    def forward(self, x, feature=False, T=1, resnet_label=None, scale=None, keep=None, ss=False, gamma_beta=None):       
        if resnet_label is not None:
            tmp_resnet_label = torch.zeros(resnet_label.shape, dtype=torch.long)
            tmp_resnet_label = tmp_resnet_label.to(resnet_label.device) + resnet_label.type(torch.long)
            self.change_fw_label(tmp_resnet_label)
            
        x = self.conv_1_3x3(x)
        x_0 = F.relu(self.bn_1(x), inplace=True)
        
        x_1 = self.stage_1(x_0)
        x_2 = self.stage_2(x_1)
        #x_3 = self.stage_3(x_2)
        #x = self.avgpool(x_3)
        #x = x.view(x.size(0), -1)
        
        if gamma_beta is not None:
            x = self.stage_3[0](x_2)
            #x = self.stage_3[0](x, gamma_beta[:,-128:])
            x = self.stage_3[1](x)
            #x = self.stage_3[1](x, gamma_beta[:,3*128:-128])
            x = self.stage_3[2](x)
            #x = self.stage_3[2](x, gamma_beta[:,2*128:3*128])
            x = self.stage_3[3](x)
            #x = self.stage_3[3](x, gamma_beta[:,128:2*128])
            x = self.stage_3[4](x, gamma_beta[:,:128])
        else:
            x = self.stage_3(x_2)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
    
        if ss is True:
            return x, x_0, x_1, x_2
        return x
    

    def forwardFeature(self, x):
        pass
                

def resnet32(**kwargs):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, **kwargs)
    return model


