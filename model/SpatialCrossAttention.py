import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super(SpatialCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, input_sequence_encoder, spatial_information_encoder):
        b, c, h, w = input_sequence_encoder.shape
        
        q = self.q_proj(input_sequence_encoder).view(b, c, -1).permute(0, 2, 1)   
        k = self.k_proj(spatial_information_encoder).view(b, c, -1).permute(0, 2, 1)   
        v = self.v_proj(spatial_information_encoder).view(b, c, -1).permute(0, 2, 1)   

        attn_output, _ = self.mha(q, k, v)   
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)  

        attn_output = self.norm1(attn_output.view(b, c, -1).permute(0, 2, 1))   
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)   

        ff_output = self.ff(attn_output.view(b, c, -1).permute(0, 2, 1))   
        ff_output = ff_output.permute(0, 2, 1).view(b, c, h, w)   

        output = self.norm2(ff_output.view(b, c, -1).permute(0, 2, 1))   
        output = output.permute(0, 2, 1).view(b, c, h, w)   

        output = output + attn_output
        return output
