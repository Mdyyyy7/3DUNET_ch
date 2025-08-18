from torch import nn
import torch
from layer import Convolution3DCH



# 3D-UNIT 神经网络encoder
class ConvBlock3D(nn.Module):
  def __init__(self, in_channels, out_channels, cross_hair, bottleneck = False):
    super(ConvBlock3D, self).__init__()
    Conv = Convolution3DCH if cross_hair else nn.Conv3d

    
    self.conv1 = Conv(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(5,5,5), padding=2,stride=1)
    self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
    self.conv2 = Conv(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(5,5,5), padding=2,stride=1)
    self.bn2 = nn.BatchNorm3d(num_features=out_channels)
    self.relu = nn.ReLU()
    self.bottleneck = bottleneck
    if not bottleneck:
        self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)


  def forward(self, input):
    res = self.relu(self.bn1(self.conv1(input)))
    res = self.relu(self.bn2(self.conv2(res)))
    out = None
    if not self.bottleneck:
        out = self.pooling(res)
    else:
        out = res
    return out, res
  



  # 3D-UNIT 神经网络decoder
class Decoderblock(nn.Module):
  def __init__(self, in_channels,  cross_hair, num_classes=None, res_channels=0, last_layer=False):
    super(Decoderblock, self).__init__()
    Conv = Convolution3DCH if cross_hair else nn.Conv3d
    self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm3d(num_features=in_channels//2)
    self.conv1 = Conv(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(5,5,5), padding=2,stride=1)
    self.conv2 = Conv(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(5,5,5), padding=2,stride=1)
    self.last_layer = last_layer
    if last_layer:
        self.conv3 = Conv(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1), padding=0,stride=1)


  def forward(self, input, residual=None):
    out = self.upconv1(input)
    if residual!=None:
      out = torch.cat((out, residual), 1)
    out = self.relu(self.bn(self.conv1(out)))
    out = self.relu(self.bn(self.conv2(out)))
    if self.last_layer:
      out = self.conv3(out)
    return out
  

  # 构建3D网络
class UNet3D(nn.Module):
  def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512, cross_hair=False) -> None:
    super(UNet3D, self).__init__()
    level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
    self.a_block1 = ConvBlock3D(in_channels=in_channels, out_channels=level_1_chnls,cross_hair=cross_hair)
    self.a_block2 = ConvBlock3D(in_channels=level_1_chnls, out_channels=level_2_chnls,cross_hair=cross_hair)
    self.a_block3 = ConvBlock3D(in_channels=level_2_chnls, out_channels=level_3_chnls,cross_hair=cross_hair)
    self.bottleNeck = ConvBlock3D(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True,cross_hair=cross_hair)
    self.s_block3 = Decoderblock(in_channels=bottleneck_channel, res_channels=level_3_chnls,cross_hair=cross_hair)
    self.s_block2 = Decoderblock(in_channels=level_3_chnls, res_channels=level_2_chnls,cross_hair=cross_hair)
    self.s_block1 = Decoderblock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True,cross_hair=cross_hair)


  def forward(self, input):
    #Analysis path forward feed
    out, residual_level1 = self.a_block1(input)
    out, residual_level2 = self.a_block2(out)
    out, residual_level3 = self.a_block3(out)
    out, _ = self.bottleNeck(out)

    #Synthesis path forward feed
    out = self.s_block3(out, residual_level3)
    out = self.s_block2(out, residual_level2)
    out = self.s_block1(out, residual_level1)
    return out