import torch
import torch.nn as nn
from einops import rearrange, repeat

from model.SwinTransformerEncoder import Encoder
from model.SpatialCrossAttention import SpatialCrossAttention
from model.TemporalCrossAttention import TemporalCrossAttention
from model.TimeCrossAttention import TimeCrossAttention
from model.SwinTransformerDecoder import Decoder

class FireTr(nn.Module):
    def __init__(self, cfg):
        super(FireTr, self).__init__()
        self.input_length = cfg['input_length']
        self.H, self.W = cfg['img_size']
        self.hidden_dim = cfg['Encoder']['hidden_dim']
        self.downscaling_factors = cfg['Encoder']['downscaling_factors']
        self.layers = cfg['Encoder']['layers']
        self.heads = cfg['Encoder']['heads']
        self.head_dim = cfg['Encoder']['head_dim']
        self.window_size = cfg['Encoder']['window_size']
        self.relative_pos_embedding = cfg['Encoder']['relative_pos_embedding']
        self.modal_channel = cfg['Encoder']['modal_channel']

        self.spatial_embed_dim = cfg['Spatial']['embed_dim']
        self.spatial_num_heads = cfg['Spatial']['num_heads']
        self.spatial_hidden_dim = cfg['Spatial']['hidden_dim']

        self.temporal_embed_dim = cfg['Temporal']['embed_dim']
        self.temporal_num_heads = cfg['Temporal']['num_heads']
        self.temporal_hidden_dim = cfg['Temporal']['hidden_dim']

        self.time_embed_dim = cfg['Time']['embed_dim']
        self.time_num_heads = cfg['Time']['num_heads']
        self.time_hidden_dim = cfg['Time']['hidden_dim']


        self.sequence_encoder = Encoder(input_channel = self.input_length, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

        self.spatial_encoder = Encoder(input_channel = self.modal_channel, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

        self.satellite_encoder = Encoder(input_channel = self.input_length, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

        self.spatial_crossattention = SpatialCrossAttention(embed_dim = self.spatial_embed_dim, num_heads = self.spatial_num_heads, ff_hidden_dim = self.spatial_hidden_dim)
        self.temporal_crossattention = TemporalCrossAttention(input_dim = self.input_length, embed_dim=self.temporal_embed_dim, num_heads=self.temporal_num_heads, ff_hidden_dim=self.temporal_hidden_dim, encoder_hidden_dim=self.temporal_hidden_dim)

        self.time_crossattention = TimeCrossAttention(nhidden=self.time_hidden_dim, embed_time=self.time_embed_dim, num_heads=self.time_num_heads)

        self.decoder = Decoder(input_channel = self.input_length, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

    def forward(self, input_sequence, fuel, vegetation, topography, satellite_images, weather_data, timestamps):
        B, T, H, W = input_sequence.shape

        input_sequence_encoder, input_sequence_encoder_list = self.sequence_encoder(input_sequence)
        spatial_satellite_images = satellite_images.reshape(B, -1, self.H, self.W)
        #spatial_information = torch.cat([fuel, vegetation, topography, spatial_satellite_images], dim=1)
        spatial_information = torch.cat([fuel, topography,], dim=1)
        print(spatial_information.shape)
        spatial_information_encoder, _ = self.spatial_encoder(spatial_information)
        spatial = self.spatial_crossattention(input_sequence_encoder, spatial_information_encoder)
        spatial = repeat(spatial, 'b c h w -> b t c h w', t=T)

        temporal_satellite_images = satellite_images.reshape(-1, 3, self.H, self.W)
        satellite_images_encoder, _ = self.satellite_encoder(temporal_satellite_images)
        satellite_images_encoder = rearrange(satellite_images_encoder, '(b t) c h w -> b t c h w', b = B, t = T)
        
        temporal = self.temporal_crossattention(satellite_images_encoder, weather_data)
        area_representation = spatial + temporal
        target_representation = self.time_crossattention(area_representation, timestamps)
        target = self.decoder(target_representation, input_sequence_encoder_list)

        return target


