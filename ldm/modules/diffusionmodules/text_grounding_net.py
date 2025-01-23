import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F


class PositionNet(nn.Module):

    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        """ {'in_dim': 768, 'out_dim': 768} """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(
            torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(
            torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings):
        """
            boxes: (batch_size, max_objs, 4)
            masks: (batch_size, max_objs)
            positive_embeddings: (batch_size, max_objs, 768)
        """
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (
            1 - masks) * positive_null # (B, max_objs, in_dim)
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null # (B, max_objs, position_dim)
        objs = self.linears(
            torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        # (B, max_objs, out_dim)
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs
