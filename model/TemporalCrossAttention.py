import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        bsz, tgt_len, embed_dim = q.size()
        src_len = k.size(1)

        q = self.q_proj(q).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class WxEncoder(nn.Module):
    def __init__(self, input_dim, num_heads=4, out_dim=768, hidden_dim=128):
        super(WxEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_feature = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ELU()
        )
        self.node_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.sublayer = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ELU())

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32).squeeze(-1)
        nodes = self.node_feature(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        attn_output, _ = self.node_attention(nodes, nodes, nodes, key_padding_mask=~mask.bool())
        feature = self.sublayer(attn_output)
        
        return feature

class TemporalCrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, encoder_hidden_dim):
        super(TemporalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.wx_encoder = WxEncoder(input_dim=6, num_heads=num_heads, out_dim=embed_dim, hidden_dim=encoder_hidden_dim)

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.mha = MultiheadAttention(embed_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, image_features, inputs):
        b, t, c, h, w = image_features.shape
        mask = torch.ones(b, t,).to(image_features.device)
        inputs = inputs[..., :6]
        weather_features = self.wx_encoder(inputs, mask)

        q = self.q_proj(image_features.view(b * t, c, h, w)).view(b, t, -1, c).reshape(b, t * h * w, c)
        k = self.k_proj(weather_features)
        v = self.v_proj(weather_features)

        attn_output = self.mha(q, k, v)
        attn_output = attn_output.view(b, t, h, w, c).permute(0, 1, 4, 2, 3)

        attn_output = attn_output + image_features

        attn_output = self.norm1(attn_output.view(b * t, c, h * w).transpose(1, 2)).transpose(1, 2).view(b, t, c, h, w)

        ff_output = self.ff(attn_output.view(b * t, c, h * w).transpose(1, 2)).transpose(1, 2).view(b, t, c, h, w)

        output = self.norm2(ff_output.view(b * t, c, h * w).transpose(1, 2)).transpose(1, 2).view(b, t, c, h, w)

        output = output + attn_output

        return output
