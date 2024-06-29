#pixmodel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
#G1:
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(GlobalGenerator, self).__init__()
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        mult = 1
        for i in range(4):
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
            mult *= 2
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        for i in range(4):
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(ngf * mult // 2),
                      nn.ReLU(True)]
            mult //= 2
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
        

#G2:
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_local_enhancers=1, n_blocks=3):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.model = GlobalGenerator(input_nc, output_nc, ngf, n_blocks)

        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model = [nn.Conv2d(4, ngf_global, kernel_size=7, padding=3),
                     nn.InstanceNorm2d(ngf_global),
                     nn.ReLU(True)]
            model += [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf_global * 2),
                      nn.ReLU(True)]
            for _ in range(n_blocks):
                model += [ResidualBlock(ngf_global * 2)]
            model += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(ngf_global),
                      nn.ReLU(True)]
            model += [nn.Conv2d(ngf_global, output_nc, kernel_size=7, padding=3),
                      nn.Tanh()]
            setattr(self, 'model' + str(n), nn.Sequential(*model))
    def forward(self, input):
        input_downsampled = [input]#3,256,256
        for n in range(1, self.n_local_enhancers + 1):
            input_downsampled.append(F.interpolate(input, scale_factor=1, mode='bilinear', align_corners=True))
        output_prev = self.model(input_downsampled[-1]) #256,256,1

        for n in range(self.n_local_enhancers):
            model = getattr(self, 'model' + str(self.n_local_enhancers - n))
            input_i = input_downsampled[self.n_local_enhancers - n - 1]

            output_prev = model(torch.cat([output_prev, input_i], 1))

        return output_prev  #1,256,256

#判别器：
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, 'layer' + str(i), netD)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        downsampled_input = input
        for i in range(self.num_D):
            model = getattr(self, 'layer' + str(i))
            output = model(downsampled_input)
            result.append(model(downsampled_input))
            if i != (self.num_D - 1):
                downsampled_input = self.downsample(downsampled_input)
        return result
