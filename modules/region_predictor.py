"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
from matplotlib import pyplot as plt

def svd(covar, fast=False):
    if fast:
        from torch_batch_svd import svd as fast_svd
        return fast_svd(covar)
    else:
        u, s, v = torch.svd(covar.cpu())
        s = s.to(covar.device)
        u = u.to(covar.device)
        v = v.to(covar.device)
        return u, s, v


class RegionPredictor(nn.Module):
    """
    Region estimating. Estimate affine parameters of the region.
    """

    def __init__(self, block_expansion, num_regions, num_channels, max_features,
                 num_blocks, temperature, estimate_affine=False, scale_factor=1,
                 pca_based=False, fast_svd=False, pad=3):
        super(RegionPredictor, self).__init__()
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.regions = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_regions, kernel_size=(7, 7),
                                 padding=pad)

        # FOMM-like regression based representation
        if estimate_affine and not pca_based:
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))
        else:
            self.jacobian = None

        # self.temperature = temperature
        self.temperature = 0.01
        self.scale_factor = scale_factor
        self.pca_based = pca_based
        self.fast_svd = fast_svd

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def region2affine(self, region, keypoints=None):
        shape = region.shape
        region = region.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], region.type()).unsqueeze_(0).unsqueeze_(0) # Replace meshgrid coordinate with Fourier PE?
        if keypoints is not None:
            mean = keypoints
        else:
            # TODO: Replace shift with user-defined keypoints
            mean = (region * grid).sum(dim=(2, 3)) # torch.Size([1, 10, 2])
        
        region_params = {'shift': mean} # batch_size, n_regions, 2

        if self.pca_based:
            mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
            covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
            covar = covar * region.unsqueeze(-1)
            covar = covar.sum(dim=(2, 3))
            region_params['covar'] = covar # batch_size, n_regions, 2, 2

        return region_params

    def forward(self, x, keypoints=None):
        
        if self.scale_factor != 1:
            x = self.down(x) # scale_factor: 0.25 -> img shape 64, 64

        feature_map = self.predictor(x) # batch_size, out_channels (35), 64, 64
        prediction = self.regions(feature_map) # batch_size, out_channels (10), 58, 58 (convolved)

        final_shape = prediction.shape
        region = prediction.view(final_shape[0], final_shape[1], -1)
        region = F.softmax(region / self.temperature, dim=2) # flatten across H, W then apply softmax
        region = region.view(*final_shape)
        
        region_params = self.region2affine(region, keypoints)
        region_params['heatmap'] = region

        # Regression-based estimation
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], 1, 4, final_shape[2],
                                                final_shape[3])
            region = region.unsqueeze(2)

            jacobian = region * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            region_params['affine'] = jacobian
            region_params['covar'] = torch.matmul(jacobian, jacobian.permute(0, 1, 3, 2))
        elif self.pca_based:
            covar = region_params['covar']
            shape = covar.shape
            covar = covar.view(-1, 2, 2)
            u, s, v = svd(covar, self.fast_svd)
            d = torch.diag_embed(s ** 0.5)
            sqrt = torch.matmul(u, d)
            sqrt = sqrt.view(*shape)
            region_params['affine'] = sqrt
            region_params['u'] = u
            region_params['d'] = d

        return region_params
