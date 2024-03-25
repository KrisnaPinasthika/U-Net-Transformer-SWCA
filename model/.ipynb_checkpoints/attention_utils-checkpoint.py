import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from einops import rearrange

def positional_encoding(max_len, embed_dim, device):
    # initialize a matrix angle_rads of all the angles
    angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(
        10_000, (2 * (np.arange(embed_dim)[np.newaxis, :] // 2)) / np.float32(embed_dim)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32, device=device, requires_grad=False)

class MultiheadAttention(nn.Module):
    """Some Information about MultiheadAttention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop_prob=0.0, lin_drop_prob=0.0, device='cuda'):
        super(MultiheadAttention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dims = embed_dim // num_heads
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3, bias=qkv_bias)
        self.lin = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
        self.att_drop = nn.Dropout(p=attn_drop_prob)
        self.lin_drop = nn.Dropout(p=lin_drop_prob)
        
        
    def forward(self, x):
        """
        input: 
            x = B, C, H, W
        output:
            out = B, C, H, W 
        """
        org_batch_size, org_channel, org_height, org_width = x.shape
        x = x.reshape(org_batch_size, org_channel, org_height*org_width).permute(0, 2, 1)
        batch_size, n_tokens, dim = x.shape 

        pe = positional_encoding(max_len=n_tokens, embed_dim=dim, device=self.device)
        x += pe
        # print(dim, self.embed_dim)
        if dim != self.embed_dim:
            print('[ERROR MHSA] : dim != embeddim') 
            raise ValueError
        
        # qkv = (batch_size, n_tokes, embed_dim * 3)
        qkv = self.qkv(x)
        
        # reshaped qkv = (batch_size, n_tokes, 3, num_heads, head_dims)
        # permuted qkv = (3, batch_size, num_heads, n_tokes, head_dims)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dims)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, and v = (batch_size, num_heads, n_tokes, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        qk_transposed = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dims) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = torch.softmax(qk_transposed, dim=-1) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = self.att_drop(attention_weights)
        
        weighted_avg = torch.matmul(attention_weights, v) # (batch_size, num_heads, n_tokes, head_dims)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(start_dim=2) # (batch_size, n_tokes, num_heads * head_dims)
        
        out = self.lin(weighted_avg) # (batch_size, n_tokes, embed_dim)
        out = self.lin_drop(out) # (batch_size, n_tokes, embed_dim)
        
        # embed_dim ~ channel
        out = out.reshape(org_batch_size, org_height, org_width, org_channel) # (batch_size, height, width, channel)
        out = out.permute(0, 3, 1, 2) # (batch_size, channel, height, width)
        
        return out
    
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(2, 3))
    
def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)]), 
        dtype=torch.long)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class MultiHeadCrossWindowAttention(nn.Module):
    """Some Information about MultiHeadCrossAttention"""
    def __init__(self, skip_channels, cyclic_shift, window_size, num_heads, 
                    qkv_bias=False, attn_drop_prob=0.0, lin_drop_prob=0.0, device='cuda'):
        super(MultiHeadCrossWindowAttention, self).__init__()
        self.device = device
        self.skip_channels = skip_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dims = skip_channels // num_heads 
        self.cyclic_shift = cyclic_shift
        
        if cyclic_shift:
            displacement = window_size // 2
            self.cyclic_propagate = CyclicShift(-displacement)
            self.cyclic_revert = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size=window_size, 
                                displacement=displacement,
                                upper_lower=True, 
                                left_right=False), 
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size=window_size, 
                                displacement=displacement,
                                upper_lower=False, 
                                left_right=True), 
                requires_grad=False
            )

        self.relative_indices = get_relative_distances(window_size) + window_size - 1
        self.pe = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        # self.pe = nn.Parameter(torch.randn(window_size**2, window_size**2), requires_grad=True)
        
        self.q = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.k = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.v = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        
        self.lin = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.lin_drop = nn.Dropout(lin_drop_prob)

    def forward(self, skip, x):
        """
        Args: 
            skip    : b, c, h, w
            x       : b, c, h, w
        Return:
        """
        if self.cyclic_shift:
            x = self.cyclic_propagate(x)
            skip = self.cyclic_propagate(skip)
            
        b, c, h, w = x.shape
        n_h, n_w = h//self.window_size, w//self.window_size
        window_squared = self.window_size*self.window_size
        
        # Reshape x and skip to [b, num_head, n_h*n_w, windows*window, head_dim]
        # print(x.shape)
        x = x.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        x = x.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        x = x.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        skip = skip.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        skip = skip.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        skip = skip.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        q = self.q(x)       # b, num_head, n_h*n_w, window_squared, head_dim
        k = self.k(x)       # b, num_head, n_h*n_w, window_squared, head_dim
        v = self.v(skip)    # b, num_head, n_h*n_w, window_squared, head_dim
        
        # qk = b, num_head, n_h*n_w, window_squared, window_squared
        qk = ( torch.matmul(q, k.transpose(3, 4)) ) / np.sqrt(self.head_dims)
        qk += self.pe[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        
        if self.cyclic_shift:
            qk[:, :, -n_w:] += self.upper_lower_mask
            qk[:, :, n_w-1::n_w] += self.left_right_mask
        
        attn_weight = self.attn_drop(torch.softmax(qk, dim=-1)) # b, num_head, n_h*n_w, window_squared, window_squared
        out = torch.matmul(attn_weight, v)  # b, num_head, n_h*n_w, window_squared, head_dim
        out = self.lin_drop(self.lin(out))  # b, num_head, n_h*n_w, window_squared, head_dim
        
        # out ==> [b, num_head, n_h*n_w, window_squared, head_dim] to [b, e, h, w]
        out = out.permute(0, 1, 4, 2, 3).reshape(b, c, n_h, n_w, self.window_size, self.window_size)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        
        if self.cyclic_shift:
            out = self.cyclic_revert(out)
        
        return out
    
if __name__ == '__main__': 
    skip, x = torch.randn((5, 32, 24, 24)).to('cpu'), torch.randn((5, 32, 24, 24)).to('cpu')
    model = MultiHeadCrossWindowAttention(
        skip_channels=32, 
        cyclic_shift=True, 
        window_size=8, 
        num_heads=2, 
        qkv_bias=False,
        attn_drop_prob=0.0, 
        lin_drop_prob=0.0, 
        device='cpu'
    ).to('cpu')
    
    print(model(skip, x).shape)