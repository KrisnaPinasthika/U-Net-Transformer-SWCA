import torch 
import torch.nn as nn
from torchvision.models import (
    efficientnet_b1, efficientnet_b5, densenet169, resnet50, resnet101, 
    EfficientNet_B1_Weights, EfficientNet_B5_Weights, DenseNet169_Weights, ResNet50_Weights, ResNet101_Weights
)

class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        self.backbone = backbone
        self.model = self.getBackbone(backbone)
        
    def forward(self, x):
        features = [x]
        
        if self.backbone[:3] == 'res': 
            encoder = list(self.model.children())
            encoder = torch.nn.Sequential(*(list(encoder)[:-2]))
        else:
            encoder = self.model.features
        
        for layer in encoder:
            features.append(layer(features[-1]))
        
        return features
    
    def getBackbone(self, backbone):
        model = {
            'eff_b1': efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1),
            'eff_b5': efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1),
            'dense_169': densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1),
            'res_50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
            'res_101': resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
        }
        
        return model[backbone]

class ResidualConv(nn.Module):
    """Some Information about ResidualConv"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.act = nn.LeakyReLU(0.2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
        )
        self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding='same', bias=False)

    def forward(self, x):
        res_x = self.res_conv(x)
        x = self.conv(x)
        x = self.act(x + res_x)

        return x

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self, skip_channel, x_channel):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = ResidualConv(in_channels=skip_channel+x_channel, out_channels=skip_channel)
        
    def forward(self, skip, x):
        """
        args: 
            skip    : (B, C, H, W)
            x       : (B, 2C, H/2, W/2)
        """
        x = self.upsample(x)                # (B, C, H, W)
        out = torch.cat([skip, x], dim=1)   # (B, 2C, H, W)
        out = self.out(out)                 # (B, C, H, W)

        return out


class UNet(nn.Module):
    
    def __init__(self, backbone:str):
        super(UNet, self).__init__()
        backbone = backbone.lower()
        self.enc = Encoder(backbone)
        self.enc_idx, channels = self.getFeatures(backbone)
        
        self.dec = nn.ModuleList([
            Decoder(skip_channel=channels[-2], x_channel=channels[-1]),
            Decoder(skip_channel=channels[-3], x_channel=channels[-2]),
            Decoder(skip_channel=channels[-4], x_channel=channels[-3]),
            Decoder(skip_channel=channels[-5], x_channel=channels[-4]),
        ])
        
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            ResidualConv(in_channels=channels[-5], out_channels=channels[-5]//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=channels[-5]//2, out_channels=1, kernel_size=1, stride=1, padding='same')
        )
        

    def forward(self, x):
        enc = self.enc(x)
        
        block1 = enc[self.enc_idx[0]]
        block2 = enc[self.enc_idx[1]]
        block3 = enc[self.enc_idx[2]]
        block4 = enc[self.enc_idx[3]]
        block5 = enc[self.enc_idx[4]]
        
        u1 = self.dec[0](block4, block5)
        u2 = self.dec[1](block3, u1)
        u3 = self.dec[2](block2, u2)
        u4 = self.dec[3](block1, u3)
        
        head = self.head(u4)
        
        return head

    def getFeatures(self, backbone): 
        channels = {
            'eff_b1': [16, 24, 40, 112, 1280],
            'eff_b5': [24, 40, 64, 176, 2048],
            'dense_169': [64, 256, 512, 1280, 1664],
            'res_50': [64, 256, 512, 1024, 2048],
            'res_101': [64, 256, 512, 1024, 2048],
        }
        
        idx_channels = {
            'eff_b1': [2, 3, 4, 6, 9],
            'eff_b5': [2, 3, 4, 6, 9],
            'dense_169': [3, 5, 7, 9, 12],
            'res_50': [3, 5, 6, 7, 8],
            'res_101': [3, 5, 6, 7, 8],
        }
        
        return idx_channels[backbone], channels[backbone]

if __name__ == '__main__': 
    device = torch.device('cpu')
    img = torch.randn(5, 3, 256, 256)
    attention_type = 'weighted'
    model = UNet(backbone='eff_b1')
    
    print(model(img).shape)