import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1)

    def forward(self, x):
        se = self.global_avg_pool(x)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DoubleUNet, self).__init__()
        self.encoder1 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.aspp1 = ASPP(512, 256)
        self.decoder1 = self.create_decoder(256, [512, 256, 128, 64])

        self.encoder2 = self.create_encoder(64, [64, 128, 256, 512])
        self.aspp2 = ASPP(512, 256)
        self.decoder2 = self.create_decoder(256, [512, 256, 128, 64], final_out_channels=out_channels)

    def create_encoder(self, in_channels, filters):
        layers = []
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SqueezeExciteBlock(out_channels))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def create_decoder(self, in_channels, filters, final_out_channels=1):
        layers = []
        for out_channels in filters:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SqueezeExciteBlock(out_channels))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, final_out_channels, kernel_size=1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder 1
        skips = []
        for i in range(23):  # VGG-19 has 23 layers in features
            x = self.encoder1[i](x)
            if i in {3, 8, 17, 26}:  # Store skip connections
                skips.append(x)

        x1 = self.aspp1(x)
        x1 = self.decoder1(x1)
        
        # Multiply input with output1
        x1 = x * x1

        # Encoder 2
        skips = []
        for i in range(0, len(self.encoder2), 2):  # Assuming 2 layers per block in encoder2
            x1 = self.encoder2[i](x1)
            x1 = self.encoder2[i + 1](x1)
            skips.append(x1)

        x2 = self.aspp2(x1)
        x2 = self.decoder2(x2)
        
        return x2


