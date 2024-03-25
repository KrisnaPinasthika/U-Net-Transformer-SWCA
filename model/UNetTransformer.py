import torch
import torch.nn as nn
import numpy as np

from torchvision.models import (
    efficientnet_b1, efficientnet_b5, resnet50, 
    EfficientNet_B1_Weights, EfficientNet_B5_Weights, ResNet50_Weights
)

class DoubleConv(nn.Module):
    """Some Information about DoubleConv"""
    def __init__(self, input_channel, output_channel, kernel_size):
        super(DoubleConv, self).__init__()
        alpha = .2
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, 
                        stride=1, padding='same', bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, 
                        stride=1, padding='same', bias=False)
        
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone):
        super(EncoderBlock, self).__init__()
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
            'res_50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        }
        
        return model[backbone]

def positional_encoding(max_len, embed_dim, device):
    # initialize a matrix angle_rads of all the angles 
    angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(10_000, (2 * (np.arange(embed_dim)[np.newaxis, :]//2)) / np.float32(embed_dim))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.tensor(pos_encoding, dtype=torch.float32, device=device)

class MultiHeadAttention(nn.Module):
    """Some Information about MultiHeadAttention"""
    def __init__(self, embed_dim, num_heads, pe_device, qkv_bias=False, att_drop_prob=.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dims = self.embed_dim // self.num_heads
        self.pe_device = pe_device
        
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3, bias=qkv_bias)    
        self.att_drop = nn.Dropout(p=att_drop_prob)
        
    def forward(self, x):
        """
        input: 
            x [query, key, value] : (N, C, H, W)
        return: 
            out : (N, C, H, W)
        """
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).permute(0, 2, 1) # (N, H*W, C)
        pe = positional_encoding(max_len=H*W, embed_dim=C, device=self.pe_device)
        x += pe # (N, H*W, C)
        
        qkv = self.qkv(x) # (N, H*W, C*3)
        qkv = qkv.reshape(N, H*W, 3, self.num_heads, self.head_dims) # (N, H*W, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, N, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        qk = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dims) # (N, num_heads, H*W, H*W)
        attention_weight = torch.softmax(qk, dim=-1) # (N, num_heads, H*W, H*W)
        attention_weight = self.att_drop(attention_weight)  # (N, num_heads, H*W, H*W)
        
        weighted_avg = torch.matmul(attention_weight, v) # (N, num_heads, H*W, head_dims)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(start_dim=2) # (N, H*W, C)
        
        out = weighted_avg.transpose(1, 2).reshape(N, C, H, W) # (N, C, H, W)
        
        return out

class MultiHeadCrossAttention(nn.Module):
    """Some Information about MultiHeadAttention
    S : input from skip connection U-Net
    Y : input from higher level ~ AKA ~ bottom up
    """
    def __init__(self, channelS, channelY, num_heads, pe_device, qkv_bias=False, att_drop_prob=.0):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = channelS
        self.num_heads = num_heads
        self.head_dims = self.embed_dim // self.num_heads
        self.pe_device = pe_device
        
        self.convS = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=channelS, 
                out_channels=channelS, 
                kernel_size=1, 
                stride=1, 
                padding='same', 
                bias=False),
            nn.BatchNorm2d(num_features=channelS), 
            nn.LeakyReLU(0.2, inplace=True) 
        )
        
        self.convY = nn.Sequential(
            nn.Conv2d(
                in_channels=channelY, 
                out_channels=channelS, 
                kernel_size=1, 
                stride=1, 
                padding='same', 
                bias=False),
            nn.BatchNorm2d(num_features=channelS), 
            nn.LeakyReLU(0.2, inplace=True) 
        )
        transpose_kernel = 2
        self.z = nn.Sequential(
            nn.Conv2d(
                in_channels=channelS, 
                out_channels=channelS, 
                kernel_size=1, 
                stride=1, 
                padding='same', 
                bias=False),
            nn.BatchNorm2d(num_features=channelS), 
            nn.Sigmoid(), 
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ConvTranspose2d(
            #     in_channels=channelS,
            #     out_channels=channelS,
            #     kernel_size=transpose_kernel,
            #     stride=2,
            #     padding=(transpose_kernel - 1) // 2,
            #     dilation=1,
            #     bias=True)
        )
        
        self.postprocessY = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ConvTranspose2d(
            #     in_channels=channelY,
            #     out_channels=channelY,
            #     kernel_size=transpose_kernel,
            #     stride=2,
            #     padding=(transpose_kernel - 1) // 2,
            #     dilation=1,
            #     bias=True), 
            nn.Conv2d(
                in_channels=channelY, 
                out_channels=channelY, 
                kernel_size=3, 
                stride=1, 
                padding='same', 
                bias=True),
            nn.Conv2d(
                in_channels=channelY, 
                out_channels=channelS, 
                kernel_size=1, 
                stride=1, 
                padding='same', 
                bias=False),
            nn.BatchNorm2d(num_features=channelS), 
            nn.LeakyReLU(0.2, inplace=True) 
        )
        
        # MHA
        self.qk = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim * 2, bias=qkv_bias)
        self.v =  nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        
        self.att_drop = nn.Dropout(p=att_drop_prob)
        
        
    def forward(self, s, y):
        """
        input: 
            s : (N, C, H, W)
            y : (N, 2C, H/2, W/2) [shape relative to s] 
        return: 
            x : (N, C/2, 2*H, 2*W)
        """
        # Todo: Add positional encoding
        ns, cs, hs, ws = s.shape
        ny, cy, hy, wy = y.shape
        s_pe = positional_encoding(max_len=hs*ws, embed_dim=cs, device=self.pe_device) # (1, H*W, C)
        y_pe = positional_encoding(max_len=hy*wy, embed_dim=cy, device=self.pe_device) # (1, [H/2]*[W/2], 2C)
        
        s = s.reshape(ns, cs, hs*ws).permute(0, 2, 1) # (N, H*W, C)
        y = y.reshape(ny, cy, hy*wy).permute(0, 2, 1) # (N, [H/2]*[W/2], 2C)
        s_new = s + s_pe # (N, H*W, C)
        y_new = y + y_pe # (N, [H/2]*[W/2], 2C)
        
        # Todo: reshape S and Y to Original Shape
        s_new = s_new.permute(0, 1, 2).reshape(ns, cs, hs, ws) # (N, C, H, W)
        y_new = y_new.permute(0, 1, 2).reshape(ny, cy, hy, wy) # (N, 2C, H/2, W/2)
        
        # Todo: Calculate MHSA
        convS = self.convS(s_new)   # (N, C, H/2, W/2)
        convY = self.convY(y_new)   # (N, C, H/2, W/2)
        new_n, new_c, new_h, new_w = convS.shape
        
        # reshape v    | from (N, C, H/2, W/2) -> (N, [H/2]*[W/2], num_heads, head_dims)
        convS = convS.flatten(start_dim=2).transpose(1, 2).reshape(new_n, new_h*new_w, self.num_heads, self.head_dims)
        # reshape q, k | from (N, C, H/2, W/2) -> (N, [H/2]*[W/2], embed_dims)
        convY = convY.flatten(start_dim=2).transpose(1, 2)
        
        # permute q, k, v => (N, num_heads, [H/2]*[W/2], head_dims)
        qk = self.qk(convY) # (N, [H/2]*[W/2], embed_dims * 2)
        qk = qk.reshape(new_n, new_h*new_w, 2, self.num_heads, self.head_dims) # (N, [H/2]*[W/2], 2, num_heads, head_dims)
        qk = qk.permute(2, 0, 3, 1, 4)  # (2, N, num_heads, [H/2]*[W/2], head_dims)
        
        # calculate q, k, v
        q, k = qk[0], qk[1]
        value = convS.permute(0, 2, 1, 3) # (N, num_heads, [H/2]*[W/2], head_dims)
        v = self.v(value)   # (N, num_heads, [H/2]*[W/2], head_dims)
        qk = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dims) # (N, num_heads, [H/2]*[W/2], [H/2]*[W/2])
        attention_weight = self.att_drop(torch.softmax(qk, dim=-1))  # (N, num_heads, [H/2]*[W/2], [H/2]*[W/2])
        
        out_mhsa = torch.matmul(attention_weight, v) # (N, num_heads, [H/2]*[W/2], head_dims)
        out_mhsa = out_mhsa.transpose(2, 3).flatten(start_dim=1, end_dim=2) # (N, [H/2]*[W/2], embed_dim)
        out_mhsa = out_mhsa.reshape(new_n, new_c, new_h, new_w) # (N, C, H/2, W/2)
    
        # Todo: calculate MHCA
        z = self.z(out_mhsa) # (N, C, H, W)
        z_out = z + s_new # (N, C, H, W)
        
        post_y = self.postprocessY(y_new) # (N, C, H, W)
        out = torch.cat([z_out, post_y], dim=1) # (N, 2C, H, W)
        
        return out

class DecoderTransformer(nn.Module):
    """Some Information about DecoderTransformer"""
    def __init__(self, channelS, channelY, num_heads, pe_device, qkv_bias=False, att_drop_prob=.0):
        super(DecoderTransformer, self).__init__()
        self.mhca = MultiHeadCrossAttention(
            channelS=channelS, 
            channelY=channelY, 
            num_heads=num_heads, 
            pe_device=pe_device, 
            qkv_bias=qkv_bias, 
            att_drop_prob=att_drop_prob, 
        )
        self.postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=channelS*2, 
                out_channels=channelS, 
                kernel_size=3, 
                stride=1, 
                padding='same', 
                bias=False),
            nn.BatchNorm2d(num_features=channelS), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=channelS, 
                out_channels=channelS, 
                kernel_size=3, 
                stride=1, 
                padding='same', 
                bias=False),
            nn.BatchNorm2d(num_features=channelS), 
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, s, y):
        x = self.mhca(s, y)
        x = self.postprocess(x)
        return x


class UNetTransformer(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, backbone, device='cuda', qkv_bias=False, att_drop_prob=.0):
        super(UNetTransformer, self).__init__()
        num_heads = 8
        
        self.encoder = EncoderBlock(backbone=backbone).to(device)
        f, self.idx = self.getFeatures(backbone)

        self.mha = MultiHeadAttention(
            embed_dim=f[-1], 
            num_heads=num_heads,
            pe_device=device,
            qkv_bias=qkv_bias, 
            att_drop_prob=att_drop_prob,
        ).to(device) # (N, C, H, W)

        self.decoder = nn.ModuleList([
            DecoderTransformer(channelS=f[3], channelY=f[4], num_heads=num_heads, pe_device=device, 
                                qkv_bias=qkv_bias, att_drop_prob=att_drop_prob), 
            DecoderTransformer(channelS=f[2], channelY=f[3], num_heads=num_heads, pe_device=device, 
                                qkv_bias=qkv_bias, att_drop_prob=att_drop_prob), 
            DecoderTransformer(channelS=f[1], channelY=f[2], num_heads=num_heads, pe_device=device, 
                                qkv_bias=qkv_bias, att_drop_prob=att_drop_prob), 
            DecoderTransformer(channelS=f[0], channelY=f[1], num_heads=num_heads, pe_device=device, 
                                qkv_bias=qkv_bias, att_drop_prob=att_drop_prob),                     
        ]).to(device)
        
        # Todo: head layer output 1 channel
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=f[0], out_channels=1, kernel_size=1, stride=1, padding="same"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Sigmoid(),
        ).to(device)

    def getFeatures(self, backbone): 
        channels = {
            'eff_b1': [16, 24, 40, 112, 1280],
            'eff_b5': [24, 40, 64, 176, 2048],
            'res_50': [64, 256, 512, 1024, 2048],
        }
        
        idx_channels = {
            'eff_b1': [2, 3, 4, 6, 9],
            'eff_b5': [2, 3, 4, 6, 9],
            'res_50': [3, 5, 6, 7, 8],
        }
        
#         channels = {
#             'eff_b1': [16, 24, 40, 80],
#             'eff_b5': [24, 40, 64, 128],
#             'res_50': [64, 256, 512, 1024],
#         }
        
#         idx_channels = {
#             'eff_b1': [2, 3, 4, 5],
#             'eff_b5': [2, 3, 4, 5],
#             'res_50': [3, 5, 6, 7],
#         }
        
        return channels[backbone], idx_channels[backbone]
    
    def forward(self, x):
        encoder = self.encoder(x)
        enc1 = encoder[self.idx[0]].clone() 
        enc2 = encoder[self.idx[1]].clone()
        enc3 = encoder[self.idx[2]].clone()
        enc4 = encoder[self.idx[3]].clone()
        enc5 = encoder[self.idx[4]].clone()
        
        
        msa = self.mha(enc5)
        
        u1 = self.decoder[0](enc4, msa)
        u2 = self.decoder[1](enc3, u1)
        u3 = self.decoder[2](enc2, u2)
        u4 = self.decoder[3](enc1, u3)
        
        return self.head(u4)
    
if __name__ == '__main__': 
    img = torch.randn((1, 3, 256, 256)).to('cuda')
    ut = UNetTransformer(backbone='res_50').to('cuda')
    
    print(ut(img).shape)
    
    # from torchsummary import summary 
    # ut = nn.UpsamplingBilinear2d(scale_factor=2)
    # print(ut(img).shape)
    # print(summary(ut, (3, 128, 128)))