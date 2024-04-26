from Blocks import *
import torch.nn.init as init
import pdb
import math
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

# 输出一下cycleGAN的判别器结构 跟IVD-NET没关系 可以随意删
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    batch_size = 2
    num_classes = 1
    # ngf = 32
    initial_kernels = 32

    input_shape = (1, 256, 256)  # c*h*w
    net = Discriminator(input_shape)
    print('model_summary', summary(net, input_size=(1, 256, 256), batch_size=10, device='cpu'))
    print("ok!")

    # net = LSTM_MMUnet(1, num_classes, ngf=ngf, temporal=3)
    # print("total parameter:" + str(netSize(net)))   # 2860,3315
    MRI = torch.randn(2, 4, 64, 64)  # bz * temporal * modal * W * H

    mmout, predict = net(MRI)
    print(mmout.shape)
    print(predict.shape)  # (2, 3, 5, 64, 64)