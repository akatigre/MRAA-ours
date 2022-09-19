
"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
import yaml
from argparse import ArgumentParser

import torch
from sklearn.model_selection import train_test_split

from scipy.spatial import ConvexHull
import numpy as np
from functools import partial
from frames_dataset import FramesDataset
from torch.utils.data import DataLoader
from skimage import io
from skimage.transform import resize

from pathlib import Path
from modules.generator import Generator
from modules.bg_motion_predictor import BGMotionPredictor
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork

def get_animation_region_params(source_region_params, driving_region_params, driving_region_params_initial,
                                mode='standard', avd_network=None, adapt_movement_scale=True):
    assert mode in ['standard', 'relative', 'avd']
    new_region_params = {k: v for k, v in driving_region_params.items()}
    if mode == 'standard':
        return new_region_params
    elif mode == 'relative':
        source_area = ConvexHull(source_region_params['shift'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(driving_region_params_initial['shift'][0].data.cpu().numpy()).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        shift_diff = (driving_region_params['shift'] - driving_region_params_initial['shift'])
        shift_diff *= movement_scale
        new_region_params['shift'] = shift_diff + source_region_params['shift']

        affine_diff = torch.matmul(driving_region_params['affine'],
                                   torch.inverse(driving_region_params_initial['affine']))
        new_region_params['affine'] = torch.matmul(affine_diff, source_region_params['affine'])
        return new_region_params
    elif mode == 'avd':
        new_region_params = avd_network(source_region_params, driving_region_params)
    return new_region_params


def iterate_dataset(root_dir, frame_shape=(256, 256, 3), is_train=True):
    """ Return two frames from same video with frame spaces by 16 timesteps
        The source is always fixed as the first frame
    """
    root_dir = Path(root_dir)
    
    if os.path.exists(root_dir/'train') and os.path.exists(root_dir/'test'):
        print('Use user-defined train-test split')
        train_videos = os.listdir(root_dir/'train')
        test_videos = os.listdir(os.path.join(root_dir, 'test'))
        root_dir = root_dir / 'train' if is_train else root_dir / 'test'
    else:
        videos = os.listdir(root_dir)
        print("Use random train-test split.")
        train_videos, test_videos = train_test_split(videos, test_size=0.2)
    
    videos = train_videos if is_train else test_videos
    resize_fn = partial(resize, output_shape=frame_shape)
    skip_frame = 16
    for video_path in sorted(videos):
        frames = [frame for frame in os.listdir(root_dir/video_path) if frame.endswith('png')]
        video_dir = root_dir/video_path
        source = resize_fn(io.imread(video_dir/frames[0]))
        out = {}
        for frame in sorted(frames)[skip_frame-1::skip_frame]:
            driving = resize_fn(io.imread(video_dir/frame))
            out['source'] = source.transpose((2, 0, 1))
            out['driving'] = driving.transpose((2, 0, 1))
            out['frame_name'] = frame
#TODO
#! Change to output OM & OF + add loss for OM and OF + NID architecture integrated
def extract(config, is_train=True):
    train_params = config['train_params']

    optimizer = torch.optim.Adam(list(generator.parameters()) +
                                 list(region_predictor.parameters()) +
                                 list(bg_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, region_predictor, bg_predictor, None,
                                      optimizer, None)
    else:
        start_epoch = 0

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params)

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        pbar = trange(start_epoch, train_params['num_epochs'])
        for epoch in pbar:
            for x in dataloader:
                losses, generated = model(x)
                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
                logger.log_iter(losses=losses)
                pbar.set_description(" ".join([f"{key}: {value:.2f} " for key, value in losses.items()]))

            scheduler.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'bg_predictor': bg_predictor,
                                     'region_predictor': region_predictor,
                                     'optimizer_reconstruction': optimizer}, inp=x, out=generated)
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--is_train", action="store_true")
    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    is_train = True if opt.mode=='train' else False

    dataset_params = config['dataset_params']
    train_params = config['train_params']
    frame_shape = dataset_params['frame_shape']
    frame_shape = (256,256,3) if not frame_shape else frame_shape

    
    # dataset = iterate_dataset(root_dir=dataset_params['root_dir'], frame_shape=frame_shape, is_train=is_train)
    #! Load Dataset
    dataset = FramesDataset(is_train=is_train, **dataset_params)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    #! Load models
    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_channels=config['model_params']['num_channels'],
                          revert_axis_swap=config['model_params']['revert_axis_swap'],
                          **config['model_params']['generator_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])

    if torch.cuda.is_available():
        region_predictor.to(opt.device_ids[0])

    if opt.verbose:
        print(region_predictor)

    bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                     **config['model_params']['bg_predictor_params'])
    if torch.cuda.is_available():
        bg_predictor.to(opt.device_ids[0])
    if opt.verbose:
        print(bg_predictor)

    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if torch.cuda.is_available():
        avd_network.to(opt.device_ids[0])
    if opt.verbose:
        print(avd_network)

    
    for x in dataloader:
        exit()
    # extract(config, is_train=opt.is_train)