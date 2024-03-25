import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_utils import (
    MultiheadAttention, 
    MultiHeadCrossWindowAttention, 
    positional_encoding
)

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""
    # Encoder : Resnet-34 like 
    def __init__(self, features, drop_rate):
        super(EncoderBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        inplace = False
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=features[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=features[0]),
            nn.LeakyReLU(0.2, inplace=inplace),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=features[0], out_channels=features[1], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[1]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[1], out_channels=features[1], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[1]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[1], out_channels=features[1], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[1]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[1], out_channels=features[1], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[1]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Dropout(drop_rate, inplace=True),

        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=features[1], out_channels=features[2], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[2]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[2], out_channels=features[2], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[2]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[2], out_channels=features[2], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[2]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[2], out_channels=features[2], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[2]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Dropout(drop_rate, inplace=True),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=features[2], out_channels=features[3], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[3]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[3]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[3]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[3]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[3]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[3]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Dropout(drop_rate, inplace=True),
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=features[3], out_channels=features[4], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[4]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[4], out_channels=features[4], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[4]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[4], out_channels=features[4], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[4]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Conv2d(in_channels=features[4], out_channels=features[4], kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(num_features=features[4]),
            nn.LeakyReLU(0.2, inplace=inplace),
            
            nn.Dropout(drop_rate, inplace=True),
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.pool(self.block2(block1))
        block3 = self.pool(self.block3(block2))
        block4 = self.pool(self.block4(block3))
        block5 = self.pool(self.block5(block4))
        
        return block1, block2, block3, block4, block5

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock"""

    def __init__(self, x_channels, skip_channels, layer, window_size, num_heads, 
                    qkv_bias, attn_drop_prob, lin_drop_prob, device):
        super(DecoderBLock, self).__init__()
        self.device = device
        self.attentions = nn.ModuleList()
        for _ in range(layer // 2):
            self.attentions.append(nn.ModuleList([
                MultiHeadCrossWindowAttention(
                    skip_channels=skip_channels, cyclic_shift=False, window_size=window_size, num_heads=num_heads, 
                    qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, lin_drop_prob=lin_drop_prob, device=device
                ),
                MultiHeadCrossWindowAttention(
                    skip_channels=skip_channels, cyclic_shift=True, window_size=window_size, num_heads=num_heads, 
                    qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, lin_drop_prob=lin_drop_prob, device=device
                ),
            ]))
        
        self.x_feed2msa = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=skip_channels, stride=1, kernel_size=1, padding='same'), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.skip_feed2msa = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=skip_channels, out_channels=skip_channels, stride=1, kernel_size=1, padding='same'), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.post_msa = nn.Sequential(
            nn.Conv2d(in_channels=skip_channels, out_channels=skip_channels, stride=1, kernel_size=1, padding='same'), 
            nn.Sigmoid()
        )
        
        self.x_after_upsample =  nn.Conv2d(
            in_channels=x_channels, 
            out_channels=skip_channels, 
            stride=1, 
            kernel_size=3, 
            padding='same'
        )
        
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=skip_channels*2, out_channels=skip_channels, kernel_size=3, stride=1, padding='same'), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=skip_channels, out_channels=skip_channels, kernel_size=3, stride=1, padding='same'), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        

    def forward(self, skip, x):
        """
        Args: 
            skip    : B, C, H, W
            x       : B, 2C, H/2, W/2
        """
        skip = self.add_abs_pe(skip, self.device)    # B, C, H, W
        x = self.add_abs_pe(x, self.device)          # B, C, H/2, W/2
        
        skip_msa = self.skip_feed2msa(skip)  # B, C, H/2, W/2
        x_msa = self.x_feed2msa(x)           # B, C, H/2, W/2
        
        for regular_window, shifted_window in self.attentions:
            x_msa = regular_window(skip_msa, x_msa) # B, C, H/2, W/2
            x_msa = shifted_window(skip_msa, x_msa) # B, C, H/2, W/2
        
        post_msa = self.post_msa(x_msa) # B, C, H/2, W/2
        post_msa = F.interpolate(
                        post_msa, 
                        size=[skip.shape[2], skip.shape[3]], 
                        mode='bilinear', 
                        align_corners=True
                    ) # B, C, H, W
        
        skip = skip * post_msa
        x = F.interpolate(x, size=[skip.shape[2], skip.shape[3]], mode='bilinear', align_corners=True) # B, 2C, H, W
        x = self.x_after_upsample(x) # B, C, H, W
        out = torch.cat([skip, x], dim=1) # B, 2C, H, W
        out = self.post_process(out) # B, C, H, W
        
        return out

    def add_abs_pe(self, x, device): 
        """
        args:
            x : B, C, H/2, W/2
        return: 
            x : B, C, H/2, W/2
        """
        b, c, origin_h, origin_w = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1) # B, HW/2, C
        b, hw, c = x.shape
        x = x + positional_encoding(max_len=hw, embed_dim=c, device=device) # B, HW/2, C
        x = x.reshape(b, origin_h, origin_w, c).permute(0, 3, 1, 2) # B, C, H/2, W/2
        
        return x

class ResidualConv(nn.Module):
    """Some Information about ResidualConv"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        res_x = self.res_conv(x)
        x = self.conv(x)
        x = x + res_x

        return x
    
class VanillaUNetSWCA(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, bottleneck_head, window_sizes:list, layers:list, 
                    qkv_bias:bool, attn_drop_prob:float, lin_drop_prob:float, drop_rate=0.2, use_sigmoid=False):
        super(VanillaUNetSWCA, self).__init__()
        features = [32, 64, 128, 256, 512]
        self.encoder = EncoderBlock(features, drop_rate).to(device)
        self.use_sigmoid = use_sigmoid
        
        if (features[-1] % bottleneck_head) != 0: 
            print(f'MSA Bottleneck Heads invalid!\nFeatures[-1] : {features[-1]}, MSA bottleneck heads : {bottleneck_head}')
            return None
        
        # Todo: Pengujian head [8, 16, 32] -> Head dims : [64, 32, 16]
        self.msa = MultiheadAttention(
            embed_dim=features[-1], 
            num_heads=bottleneck_head, 
            qkv_bias=qkv_bias, 
            attn_drop_prob=attn_drop_prob, 
            lin_drop_prob=lin_drop_prob, 
            device=device)
        
        # Todo: For decoders all head dims are 16
        dec_heads = [2, 4, 8, 16]
        self.decoder = nn.ModuleList([
            DecoderBLock(
                x_channels=features[-1], skip_channels=features[-2], layer=layers[-1], num_heads=dec_heads[-1], 
                window_size=window_sizes[-1], qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, 
                lin_drop_prob=lin_drop_prob, device=device
            ),
            DecoderBLock(
                x_channels=features[-2], skip_channels=features[-3], layer=layers[-2], num_heads=dec_heads[-2], 
                window_size=window_sizes[-2], qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, 
                lin_drop_prob=lin_drop_prob, device=device
            ),
            DecoderBLock(
                x_channels=features[-3], skip_channels=features[-4], layer=layers[-3], num_heads=dec_heads[-3], 
                window_size=window_sizes[-3], qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, 
                lin_drop_prob=lin_drop_prob, device=device
            ),
            DecoderBLock(
                x_channels=features[-4], skip_channels=features[-5], layer=layers[-4], num_heads=dec_heads[-4], 
                window_size=window_sizes[-4], qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, 
                lin_drop_prob=lin_drop_prob, device=device
            ),
        ]).to(device)
        
        self.head = nn.Sequential(
            ResidualConv(in_channels=features[-5], out_channels=features[-5]//2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=features[-5]//2, out_channels=1, kernel_size=1, stride=1, padding="same"),
        ).to(device)
        
        if self.use_sigmoid == True:
            self.last_act = nn.Sigmoid()
            
    def forward(self, x):
        block1, block2, block3, block4, block5 = self.encoder(x) 
        
        msa = self.msa(block5)
        u1 = self.decoder[0](block4, msa)
        u2 = self.decoder[1](block3, u1)
        u3 = self.decoder[2](block2, u2)
        u4 = self.decoder[3](block1, u3)
        
        out = self.head(u4)
        
        if self.use_sigmoid == True:
            return self.last_act(out)

        return out

if __name__ == '__main__': 
    from torchsummary import summary
    
    img = torch.randn((2, 3, 256, 256)).to('cuda')
    model = VanillaUNetSWCA(
        device='cuda', 
        bottleneck_head=16,
        window_sizes=[8, 8, 8, 8], 
        layers=[4, 4, 4, 4], 
        qkv_bias=False, 
        attn_drop_prob=0.2, 
        lin_drop_prob=0.1,).to('cuda')
    
    print(model(img).shape)