from abc import abstractmethod
from functools import partial
import math

import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
# from .positionnet  import PositionNet
from torch.utils import checkpoint
from ldm.util import instantiate_from_config
from copy import deepcopy


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context, objs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, objs)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels, # model_channels = 320
        emb_channels, # model_channels * 4
        dropout, # 0
        out_channels=None, # mult * model_channels
        use_conv=False,
        use_scale_shift_norm=False, # False
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims,
                        self.out_channels,
                        self.out_channels,
                        3,
                        padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           3,
                                           padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return_ = self.skip_connection(x) + h
        return return_


class UNetModel(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        use_scale_shift_norm=False,
        transformer_depth=1,
        context_dim=None,
        fuser_type=None,
        inpaint_mode=False,
        grounding_downsampler=None,
        grounding_tokenizer=None,
    ):
        """
{'image_size': 64, 'in_channels': 4, 'out_channels': 4,
'model_channels': 320, 'attention_resolutions': [4, 2, 1],
'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
'num_heads': 8, 'transformer_depth': 1,
'context_dim': 768, 'fuser_type': 'gatedSA', 'use_checkpoint': True,
'grounding_tokenizer':
    {'target': 'ldm.modules.diffusionmodules.text_grounding_net.PositionNet',
    'params': {'in_dim': 768, 'out_dim': 768}}}}

        """
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.inpaint_mode = inpaint_mode
        assert fuser_type in ["gatedSA", "gatedSA2", "gatedCA"]

        self.grounding_tokenizer_input = None  # set externally
        """
config['grounding_tokenizer_input']: 
     {'target': 'grounding_input.text_grounding_tokinzer_input.GroundingNetInput'}
grounding_tokenizer_input
    <grounding_input.text_grounding_tokinzer_input.GroundingNetInput>
        """

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.downsample_net = None
        self.additional_channel_from_downsampler = 0
        self.first_conv_type = "SD"
        self.first_conv_restorable = True
        if grounding_downsampler is not None:
            self.downsample_net = instantiate_from_config(grounding_downsampler)
            self.additional_channel_from_downsampler = self.downsample_net.out_dim
            self.first_conv_type = "GLIGEN"

        if inpaint_mode:
            # The new added channels are: masked image (encoded image) and mask, which is 4+1
            in_c = in_channels + self.additional_channel_from_downsampler + in_channels + 1
            self.first_conv_restorable = False  # in inpaint; You must use extra channels to take in masked real image
        else:
            in_c = in_channels + self.additional_channel_from_downsampler
        """
        dims = 2
        in_c = in_channels = 4
        model_channels = 320
        """
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_c, model_channels, 3, padding=1)) # 3: kernel_size ,
        ])

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # = = = = = = = = = = = = = = = = = = = = Down Branch = = = = = = = = = = = = = = = = = = = = #
        for level, mult in enumerate(channel_mult): # channel_mult: [1, 2, 4, 4]
            for _ in range(num_res_blocks): # 2
                layers = [
                    ResBlock(
                        ch, # model_channels = 320
                        time_embed_dim, # model_channels * 4
                        dropout, # 0
                        out_channels=mult * model_channels,
                        dims=dims, # 2
                        use_checkpoint=use_checkpoint, # True
                        use_scale_shift_norm=use_scale_shift_norm, # False
                    )
                ]

                ch = mult * model_channels
                # ds: 1 -> 2 -> 4 ->  8
                if ds in attention_resolutions: # attention_resolutions:[4, 2, 1]
                    dim_head = ch // num_heads # num_heads: 8
                    layers.append(
                        SpatialTransformer(ch, # mult * model_channels
                                           key_dim=context_dim, # 768
                                           value_dim=context_dim, # 768
                                           n_heads=num_heads, # 8
                                           d_head=dim_head, # ( ch // num_heads )
                                           depth=transformer_depth, # 1
                                           fuser_type=fuser_type, # gatedSA
                                           use_checkpoint=use_checkpoint) # True
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(
                    channel_mult
            ) - 1:  # will not go to this downsample branch in the last feature
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch,
                                   conv_resample, # True
                                   dims=dims, # 2
                                   out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        dim_head = ch // num_heads

        # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]

        # = = = = = = = = = = = = = = = = = = = = BottleNeck = = = = = = = = = = = = = = = = = = = = #

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch,
                               key_dim=context_dim,
                               value_dim=context_dim,
                               n_heads=num_heads,
                               d_head=dim_head,
                               depth=transformer_depth,
                               fuser_type=fuser_type,
                               use_checkpoint=use_checkpoint),
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm))

        # = = = = = = = = = = = = = = = = = = = = Up Branch = = = = = = = = = = = = = = = = = = = = #

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich,
                             time_embed_dim,
                             dropout,
                             out_channels=model_channels * mult,
                             dims=dims,
                             use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch,
                                           key_dim=context_dim,
                                           value_dim=context_dim,
                                           n_heads=num_heads,
                                           d_head=dim_head,
                                           depth=transformer_depth,
                                           fuser_type=fuser_type,
                                           use_checkpoint=use_checkpoint))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch,
                                 conv_resample,
                                 dims=dims,
                                 out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]
        # self.middle_block = [ RTR ]
        # self.output_blocks = [ R  R  RU | RT  RT  RTU |  RT  RT  RTU  |  RT  RT  RT  ]

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        """
'grounding_tokenizer':
    {'target': 'ldm.modules.diffusionmodules.text_grounding_net.PositionNet',
    'params': {'in_dim': 768, 'out_dim': 768}}}}
        """
        self.position_net = instantiate_from_config(grounding_tokenizer)

    def restore_first_conv_from_SD(self):
        """
`restore_first_conv_from_SD` 함수는 `UNetModel` 클래스의 첫 번째 컨볼루션 레이어를 "SD" (Stable Diffusion) 모델의 가중치로 복원하는 역할을 합니다. 이 함수는 다음과 같은 작업을 수행합니다:

1. `first_conv_restorable` 속성이 `True`인지 확인합니다.
2. 첫 번째 컨볼루션 레이어의 장치를 가져옵니다.
3. "SD_input_conv_weight_bias.pth" 파일에서 Stable Diffusion 모델의 가중치를 로드합니다.
4. 현재 첫 번째 컨볼루션 레이어의 상태를 `GLIGEN_first_conv_state_dict`에 백업합니다.
5. 첫 번째 컨볼루션 레이어를 새로운 `conv_nd` 레이어로 교체하고, 로드한 가중치를 적용합니다.
6. 새로운 레이어를 원래 장치로 이동시킵니다.
7. `first_conv_type` 속성을 "SD"로 설정합니다.

만약 `first_conv_restorable` 속성이 `False`인 경우, 해당 레이어가 복원 불가능하다는 메시지를 출력합니다.

        """
        if self.first_conv_restorable:
            device = self.input_blocks[0][0].weight.device

            SD_weights = th.load("SD_input_conv_weight_bias.pth")
            self.GLIGEN_first_conv_state_dict = deepcopy(
                self.input_blocks[0][0].state_dict())
            """
            nn.Conv2d
                in_c = 4
                out_c = 320
                kernel_size = 3 
            """
            self.input_blocks[0][0] = conv_nd(2, 4, 320, 3, padding=1)
            self.input_blocks[0][0].load_state_dict(SD_weights)
            self.input_blocks[0][0].to(device)

            self.first_conv_type = "SD"
        else:
            print(
                "First conv layer is not restorable and skipped this process, probably because this is an inpainting model?"
            )

    def restore_first_conv_from_GLIGEN(self):
        breakpoint()  # TODO

    def forward(self, input):
        """
    input = dict(
        x=starting_noise,grounding_input
            None
        timesteps=(batch_size) -> step 값으로 전부 채워짐,
        context=context,
            (batch_size, 77, 768)
        grounding_input=grounding_input, (DICT)
            boxes: (batch_size, max_objs, 4)
            masks: (batch_size, max_objs)
            positive_embeddings: (batch_size, max_objs, 768)
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,)

        """
        if ("grounding_input" in input):
            grounding_input = input["grounding_input"]
        else:
            # Guidance null case
            """
    config['grounding_tokenizer_input']: 
         {'target': 'grounding_input.text_grounding_tokinzer_input.GroundingNetInput'}
    grounding_tokenizer_input
        <grounding_input.text_grounding_tokinzer_input.GroundingNetInput>
            """
            grounding_input = self.grounding_tokenizer_input.get_null_input()

        if self.training and random.random(
        ) < 0.1 and self.grounding_tokenizer_input.set:  # random drop for guidance
            grounding_input = self.grounding_tokenizer_input.get_null_input()
        # Grounding tokens: B*N*C
        objs = self.position_net(**grounding_input) # (B, max_objs, out_dim)

        # Time embedding
        # t_emb = (N, model_channels)
        t_emb = timestep_embedding(input["timesteps"], # (batch_size) -> step 값으로 전부 채워짐
                                   self.model_channels, # 320
                                   repeat_only=False)
        # emb: (N , time_embed_dim = 4 * model_channels)
        emb = self.time_embed(t_emb)

        # input tensor
        h = input["x"]
        """
            # 각 요소는 평균이 0이고 표준편차가 1인 정규분포에서 무작위로 선택된 값
            # (batch_size, in_channels, image_size, image_size) = ( b, 4, 64, 64)
        """
        if self.downsample_net != None and self.first_conv_type == "GLIGEN":
            temp = self.downsample_net(input["grounding_extra_input"])
            h = th.cat([h, temp], dim=1)
        if self.inpaint_mode:
            if self.downsample_net != None:
                breakpoint()  # TODO: think about this case
            h = th.cat([h, input["inpainting_extra_input"]], dim=1)

        # Text input
        # TODO: 여기서부터
        context = input["context"] # (batch_size, 77, 768)
        # 77은 CLIP 모델에서 사용하는 텍스트 토큰의 최대 길이를 나타냄
        # CLIP 모델은 입력 텍스트를 토큰화하여 고정된 길이의 시퀀스로 변환하며,
        #     이 경우 최대 77개의 토큰으로 변환

        # Start forwarding
        hs = []
        # # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]
        for idx, module in enumerate(self.input_blocks):
            """
            h : (batch_size, in_channels, image_size, image_size)
            emb : (N , time_embed_dim = 4 * model_channels)
            context : (batch_size, 77, 768)
            objs : (B, max_objs, out_dim)
            """
            print(f"------------{idx}-----------")
            print("h.shape: ", h.shape)
            print("emb.shape: ", emb.shape)
            print("context.shape: ", context.shape)
            print("objs.shape: ", objs.shape)
            h = module(h, emb, context, objs)
            print("h.shape: ", h.shape)
            hs.append(h)
        raise NotImplementedError

        h = self.middle_block(h, emb, context, objs)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, objs)

        return self.out(h)
