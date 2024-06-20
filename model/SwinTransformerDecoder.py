from torch import nn
from model.SwinTransformer import StageModule, StageModule_up, StageModule_up_final
from einops import rearrange, repeat

class Decoder(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads, head_dim, window_size, relative_pos_embedding):
        super(Decoder, self).__init__()

        self.stage4 = StageModule_up(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 2,
                                     layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[32, 32])

        self.stage5 = StageModule_up(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 1,
                                     layers=layers[1], upscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[64, 64])

        self.stage6 = StageModule_up_final(in_channels=hidden_dim*2, hidden_dimension=1,
                                     layers=layers[0], upscaling_factor=downscaling_factors[0], num_heads=heads[0],
                                     head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[256, 256])

    def forward(self, x, res):
        B, T, _, _, _ = x.shape
        x1 = repeat(res[0], 'b c h w -> b t c h w', t=T)
        x2 = repeat(res[1], 'b c h w -> b t c h w', t=T)

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x1 = rearrange(x1, 'b t c h w -> (b t) c h w')
        x2 = rearrange(x2, 'b t c h w -> (b t) c h w')   
         
        x3 = self.stage4(x, x2) 
        x4 = self.stage5(x3, x1)                
        x5 = self.stage6(x4)
        
        x5 = rearrange(x5, '(b t) c h w -> b t c h w', b=B, t=T).squeeze()
        return x5