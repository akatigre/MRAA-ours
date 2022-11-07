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
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, SameBlock2d_DoubleConv
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from skimage.transform import resize
import liif_models
from utils import make_coord

class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """
    #! add a case where reconstructed structure is added
    def __init__(self, num_channels, num_regions, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks,  dataset=None, pixelwise_flow_predictor_params=None, skips=False, revert_axis_swap=True, liif_model_path='checkpoints/edsr-baseline-liif.pth'):
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

        # Naive Solution
    
        # down_blocks = []
        # self.first2 = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        # for i in range(num_down_blocks):
        #     in_features = min(max_features, block_expansion * (2 ** i))
        #     out_features = min(max_features, block_expansion * (2 ** (i + 1)))
        #     down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # self.down_blocks2 = nn.ModuleList(down_blocks)
        # up_blocks = []
        # for i in range(num_down_blocks):
        #     in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
        #     out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
        #     up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # self.up_blocks2 = nn.ModuleList(up_blocks)


        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        # self.final2 = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.skips = skips
        if dataset is not None:
            video_shape = (384, 384) if dataset=='ted' else (256, 256)
            self.h, self.w = video_shape
        else:
            self.h, self.w = None, None

        # With LIIF Representation
        self.liif = liif_models.make(torch.load(liif_model_path)['model'], load_sd=True)
        self.liif.eval()
        

        # liif_encode = []
        liif_decode = []
        for i in range(num_down_blocks * 2):
            # liif_encode.append(SameBlock2d_DoubleConv(block_expansion, block_expansion, kernel_size=3, padding=1))
            liif_decode.append(SameBlock2d_DoubleConv(block_expansion * 2, block_expansion, kernel_size=3, padding=1))

        # self.liif_encode = nn.ModuleList(liif_encode)
        self.liif_decode = nn.ModuleList(liif_decode)


    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow), optical_flow

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None, applyDeformation=True):
        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['optical_flow']
            
            if applyDeformation:
                input_skip, _ = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                else:
                    input_skip = input_skip * occlusion_map
                    
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out, occlusion_map

    def forward(self, source_structure, source_image, driving_region_params=None, source_region_params=None, bg_params=None):
        #! Jointly Optimizing Stage 1 & Stage 2
        # TODO Stage 1
        out = self.first(source_structure) # source_struc
        skips = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        output_dict = {}

        if self.pixelwise_flow_predictor is not None:
            
            motion_params = self.pixelwise_flow_predictor(source_image=source_structure,
                                                        driving_region_params=driving_region_params,
                                                        source_region_params=source_region_params,
                                                        bg_params=bg_params)
        output_dict["deformed_structure"], output_dict["optical_flow"] = self.deform_input(source_structure, motion_params['optical_flow'])
        out, _ = self.apply_optical(input_previous=None, input_skip=out, motion_params=motion_params)
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
            output_dict["prediction_structure"], output_dict['occlusion_map'] = self.apply_optical(input_skip=source_structure, input_previous=out, motion_params=motion_params)  
        
        # TODO Stage 2
        out = F.interpolate(out, size=source_image.shape[2:], mode='bilinear')
        source_feature = self.liif.gen_feat(source_image)
        out_feature = self.liif.gen_feat(out)
        # skips = [out]
        # for i, encode_block in enumerate(self.liif_encode):
        #     out = encode_block(out)
        #     skips.append(out)

        output_dict['deformed'], output_dict['optical_flow'] = self.deform_input(source_image, output_dict['optical_flow'])
        
        for i, decode_block in enumerate(self.liif_decode):
            if self.skips:
                out_feature, _ = self.apply_optical(input_skip=source_feature, input_previous=out_feature, motion_params=motion_params)
            out_feature = decode_block(source_feature, out_feature)
        
        if self.skips:
            out_feature, _ = self.apply_optical(input_skip=source_feature, input_previous=out_feature, motion_params=motion_params, applyDeformation=False) 
        
        h, w = source_feature.shape[2:4]
        batch = source_feature.shape[0]
        coord = make_coord((h,w)).cuda()
        cell = torch.ones_like(coord)
        cell[:,0] *= 2 / h
        cell[:,1] *= 2 / w
        coord = coord.repeat(batch, *([1] * len(cell.shape)))
        cell = cell.repeat(batch, *([1] * len(cell.shape)))
        
        ql = 0
        preds = []
        n = coord.shape[1]

        while ql < n:
            qr = min(ql + 30000, n)
            pred = self.liif.query_rgb(coord[:,ql:qr, :], cell[:, ql:qr, :], out_feature)
            preds.append(pred)
            ql = qr

        out = torch.cat(preds,dim=1)
        output_dict["prediction"] = out.clamp(0,1).view(batch, h, w, self.num_channels,).permute(0,3,1,2)
        
        return output_dict