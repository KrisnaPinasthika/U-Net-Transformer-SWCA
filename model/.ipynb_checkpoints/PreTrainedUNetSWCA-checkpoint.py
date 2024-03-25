import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights
)
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    densenet169, DenseNet169_Weights, 
    densenet201, DenseNet201_Weights
)
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

from .attention_utils import (
    MultiheadAttention, 
    MultiHeadCrossWindowAttention, 
    positional_encoding
)

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone_name, freeze=False):
        super(EncoderBlock, self).__init__()
        self.backbone_name = backbone_name
        # Backbones: 
        # EfficientNet  : B1, B3, B5, B6
        # DenseNet      : 121, 169, 201
        # ResNet        : 18, 34, 50, 101
        backbones = {
            'eff_b1' : efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1),
            'eff_b3' : efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1),
            'eff_b5' : efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1),
            'eff_b6' : efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1),
            
            'dense_121': densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
            'dense_169': densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1),
            'dense_201': densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1),
            
            'res_18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
            'res_34': resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
            'res_50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
            'res_101': resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
        }
        self.backbone = backbones.get(backbone_name)
        
        if self.backbone == None:
            print('Check your backbone again ^.^')
            return None
            
        if freeze:
            for v in self.backbone.parameters():
                v.requires_grad = False

    def forward(self, x):
        features = [x]
        
        if self.backbone_name[:3] == 'res': 
            encoder = list(self.backbone.children())
            encoder = torch.nn.Sequential(*(list(encoder)[:-2]))
        else:
            encoder = self.backbone.features
        
        for layer in encoder:
            features.append(layer(features[-1]))

        return features

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
        x = self.add_abs_pe(x, self.device)          # B, 2C, H/2, W/2
        
        skip_msa = self.skip_feed2msa(skip)  # B, C, H/2, W/2
        x_msa = self.x_feed2msa(x)           # B, C, H/2, W/2
        
        # print(f"[Decoder] skip_msa: {skip_msa.shape}, x_msa : {x_msa.shape}")
        
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

class PreTrainedUNetSWCA(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, backbone_name, bottleneck_head, window_sizes:list, layers:list, 
                    qkv_bias:bool, attn_drop_prob:float, lin_drop_prob:float):
        super(PreTrainedUNetSWCA, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.encoder = EncoderBlock(self.backbone_name, freeze=False).to(device)
        
        # EfficientNet-B1, B3, B5, B6
        # Todo: EfficientNet Attention Head dimension = 8
        if self.backbone_name == 'eff_b1':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [16, 24, 40, 112, 1280]
            dec_heads = [2, 3, 5, 14]
        elif self.backbone_name == 'eff_b3':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [24, 32, 48, 136, 1536]
            dec_heads = [3, 4, 6, 17]
        elif self.backbone_name == 'eff_b5':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [24, 40, 64, 176, 2048]
            dec_heads = [3, 5, 8, 22]
        elif self.backbone_name == 'eff_b6':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [32, 40, 72, 200, 2304]
            dec_heads = [4, 5, 9, 25]
        
        # DenseNet
        # Todo: EfficientNet Attention Head dimension = 16
        elif self.backbone_name == 'dense_121':
            self.block_idx = [3, 5, 7, 9, 12]
            features = [64, 256, 512, 1024, 1024]
            dec_heads = [4, 16, 32, 64]
        elif self.backbone_name == 'dense_169':
            self.block_idx = [3, 5, 7, 9, 12]
            features = [64, 256, 512, 1280, 1664]
            dec_heads = [4, 16, 32, 80]
        elif self.backbone_name == 'dense_201':
            self.block_idx = [3, 5, 7, 9, 12]
            features = [64, 256, 512, 1792, 1920]
            dec_heads = [4, 16, 32, 112]
            
        # ResNet
        # Todo: ResNet Attention Head dimension = 16
        elif self.backbone_name == 'res_18':
            self.block_idx = [3, 5, 6, 7, 8]
            features = [64, 64, 128, 256, 512]
            dec_heads = [4, 4, 8, 16]
        elif self.backbone_name == 'res_34':
            self.block_idx = [3, 5, 6, 7, 8]
            features = [64, 64, 128, 256, 512]
            dec_heads = [4, 4, 8, 16]
        elif self.backbone_name == 'res_50':
            self.block_idx = [3, 5, 6, 7, 8]
            features = [64, 256, 512, 1024, 2048]
            dec_heads = [4, 16, 32, 64]
        elif self.backbone_name == 'res_101':
            self.block_idx = [3, 5, 6, 7, 8]
            features = [64, 256, 512, 1024, 2048]
            dec_heads = [4, 16, 32, 64]
        
        else:
            print('Check your backbone again ^.^')
            return None
        
        if (features[-1] % bottleneck_head) != 0: 
            print(f'MSA Bottleneck Heads invalid!\nFeatures[-1] : {features[-1]}, MSA bottleneck heads : {bottleneck_head}')
            return None
        
        self.msa = MultiheadAttention(
            embed_dim=features[-1], 
            num_heads=bottleneck_head, 
            qkv_bias=qkv_bias, 
            attn_drop_prob=attn_drop_prob, 
            lin_drop_prob=lin_drop_prob, 
            device=device)
        
        # dec_heads = [8, 8, 8, 8]
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
        
    def forward(self, x):
        enc = self.encoder(x) 
        
        if self.backbone_name[:3] == 'res':
            block1 = enc[self.block_idx[0]].clone()
            block2 = enc[self.block_idx[1]].clone()
            block3 = enc[self.block_idx[2]].clone()
            block4 = enc[self.block_idx[3]].clone()
            block5 = enc[self.block_idx[4]].clone()
        else:
            block1 = enc[self.block_idx[0]]
            block2 = enc[self.block_idx[1]]
            block3 = enc[self.block_idx[2]]
            block4 = enc[self.block_idx[3]]
            block5 = enc[self.block_idx[4]]
        
#         print(f"Block 1 : {block1.shape}")
#         print(f"Block 2 : {block2.shape}")
#         print(f"Block 3 : {block3.shape}")
#         print(f"Block 4 : {block4.shape}")
#         print(f"Block 5 : {block5.shape}")
        
        msa = self.msa(block5)
        # print(f"MSA : {msa.shape}")
        
        u1 = self.decoder[0](block4, msa)
        u2 = self.decoder[1](block3, u1)
        u3 = self.decoder[2](block2, u2)
        u4 = self.decoder[3](block1, u3)
        
        # print(f"u1 : {u1.shape}")
        # print(f"u2 : {u2.shape}")
        # print(f"u3 : {u3.shape}")
        # print(f"u4 : {u4.shape}")
        
        head = self.head(u4)
        # print(f"head : {head.shape}")
        
        return head

if __name__ == '__main__': 
    from torchsummary import summary
    # print(resnet34())
    img = torch.randn((2, 3, 256, 256)).to('cuda')
    model = PreTrainedUNetSWCA(
        device='cuda', 
        backbone_name='res_34', 
        bottleneck_head=16,
        window_sizes=[8, 8, 8, 8], 
        layers=[2, 2, 2, 2], 
        qkv_bias=False, 
        attn_drop_prob=0., 
        lin_drop_prob=0.
    ).to('cuda')
    
    print(model(img).shape)