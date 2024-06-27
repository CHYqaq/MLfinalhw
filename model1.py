import torch
import torch.nn as nn
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.output = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.output(x)
        return x

class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DoubleUNet, self).__init__()
        self.encoder1 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.aspp1 = ASPP(512, 256)
        self.decoder1 = self.create_decoder(256, 128) # 修改解码器的输出通道数为 128

        self.encoder2 = self.create_encoder(128, 64)  # 修改输入通道数为 decoder1 的输出通道数
        self.aspp2 = ASPP(64, 32)
        self.decoder2 = self.create_final_decoder(32, out_channels)

    def create_encoder(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def create_decoder(self, in_channels, mid_channels):
        layers = [
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def create_final_decoder(self, in_channels, out_channels):
        layers = [
            nn.ConvTranspose2d(in_channels, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8,out_channels , kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        #print("x",x.size())
        x1 = self.encoder1(x)
        #print("x1",x1.size())
        x1 = self.aspp1(x1)
        #print("x1",x1.size())
        x1 = self.decoder1(x1)
        #print("x1",x1.size())
        x2 = self.encoder2(x1)
        #print("x2",x2.size())
        x2 = self.aspp2(x2)
        #print("x2",x2.size())
        x2 = self.decoder2(x2)
        #print("x2",x2.size())
        return x2

# model = DoubleUNet(in_channels=3, out_channels=1)
