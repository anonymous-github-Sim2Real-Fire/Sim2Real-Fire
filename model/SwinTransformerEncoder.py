from torch import nn
from einops import rearrange
from .SwinTransformer import StageModule, StageModule_up, StageModule_up_final

class Encoder(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads, head_dim, window_size, relative_pos_embedding):
        super(Encoder, self).__init__()        
        self.stage1 = StageModule(in_channels=input_channel, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[64, 64])

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[32, 32])

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[16, 16])


    def forward(self, images): 
        x1 = self.stage1(images)
        x2 = self.stage2(x1)    
        x3 = self.stage3(x2) 
        return x3, [x1, x2, x3]
    
