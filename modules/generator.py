"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from skimage.transform import resize

class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_regions, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks,  dataset=None, pixelwise_flow_predictor_params=None, skips=False, revert_axis_swap=True):
        super(Generator, self).__init__()

        if pixelwise_flow_predictor_params is not None:
            self.pixelwise_flow_predictor = PixelwiseFlowPredictor(num_regions=num_regions, num_channels=num_channels,
                                                                   revert_axis_swap=revert_axis_swap,
                                                                   **pixelwise_flow_predictor_params)
        else:
            self.pixelwise_flow_predictor = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.skips = skips
        if dataset is not None:
            video_shape = (384, 384) if dataset=='ted' else (256, 256)
            self.h, self.w = video_shape
        else:
            self.h, self.w = None, None

    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow)

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None):
        warped = {}

        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['optical_flow']
            
            input_skip = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    if input_skip.shape[1]==3:
                        warped['input_skip'] = input_skip # deformed source
                        warped['input_previous'] = input_previous # deformed & occluded from previous
                        
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                    if input_skip.shape[1]==3:
                        warped['input_occluded'] = input_skip
                else:
                    input_skip = input_skip * occlusion_map
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out, warped

    def forward(self, source_image256, source_image512, driving_region_params, source_region_params, bg_params=None):
        gt_source_shape = source_image256.shape[0]
        real_source_shape = source_image512.shape[0]
        out = self.first(source_image512) # torch.Size([1, 64, 512, 512])
        skips = [out]
        for i in range(len(self.down_blocks)):
            # torch.Size([1, 128, 256, 256])
            # torch.Size([1, 256, 128, 128])
            out = self.down_blocks[i](out)
            skips.append(out)

        output_dict = {}

        if self.pixelwise_flow_predictor is not None:
            
            motion_params = self.pixelwise_flow_predictor(source_image=source_image256,
                                                          driving_region_params=driving_region_params,
                                                          source_region_params=source_region_params,
                                                          bg_params=bg_params)
            
            
            for k in ['optical_flow', 'occlusion_map', 'sparse_motion']:
                # sparse_motion: torch.Size([1, 11, 64, 64, 2])
                # optical_flow: torch.Size([1, 64, 64, 2])
                # occlusion_map: torch.Size([1, 1, 64, 64])
                if gt_source_shape != real_source_shape:
                    if k=='optical_flow':
                        v = motion_params[k].permute(0, 3, 1, 2)
                    elif k=='sparse_motion':
                        v = motion_params[k].permute(0, 1, 4, 2, 3)
                    else:
                        v = motion_params[k]
                    interp_size = (512//4, 512//4)
                    if v.dim()!=4:
                        v_ = []
                        for idx in range(v.shape[1]):
                            v_.append(F.interpolate(v[:,idx], size=interp_size, mode='nearest'))
                        v = torch.cat(v_, dim=0)[None]
                    else:
                        v = F.interpolate(v, size=interp_size, mode='nearest')
                    if k=='optical_flow':
                        v = v.permute(0, 2, 3, 1)
                    elif k=='sparse_motion':
                        v = v.permute(0, 1, 3, 4, 2)
                    motion_params[k] = v
                
                   
                output_dict[k] = motion_params.get(k)
            output_dict["deformed"] = self.deform_input(source_image512, motion_params['optical_flow'])

        else:
            motion_params = None
        
        out, warped = self.apply_optical(input_previous=None, input_skip=out, motion_params=motion_params)
        for k, v in warped.items():
            output_dict[k] = v

        out = self.bottleneck(out)
        
        for i in range(len(self.up_blocks)):
            if self.skips:
                out, _ = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out, motion_params=motion_params)
            out = self.up_blocks[i](out)
        
        if self.skips:
            out, _ = self.apply_optical(input_skip=skips[0], input_previous=out, motion_params=motion_params)
        out = self.final(out) # B, 3, H, W
        out = F.sigmoid(out)
        if self.skips:
            out, warped = self.apply_optical(input_skip=source_image512, input_previous=out, motion_params=motion_params)

        for k, v in warped.items():
            output_dict[k] = v
        output_dict["prediction"] = out

        return output_dict
