"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
from functools import partial

from PIL import Image
import cv2

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)]
        if frame_shape is not None:
            video_array = np.array([resize(frame, frame_shape) for frame in video_array])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if frame_shape is None:
            raise ValueError('Frame shape can not be None for stacked png format.')
        
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = resize(np.moveaxis(image, 1, 0), frame_shape)
        
        video_array = video_array.reshape((-1,) + frame_shape + (3, ))
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, structure_root_dir, img_frame_shape=(256, 256, 3), structure_frame_shape=(128,128,3), structure=False, id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = img_frame_shape
        self.structure_shape = structure_frame_shape
        self.structure = structure
        if self.structure:
            self.structure_root_dir = structure_root_dir
        self.edge_filter = partial(cv2.edgePreservingFilter, flags=cv2.RECURS_FILTER, sigma_s=100, sigma_r=0.7)
        
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            if self.structure:
                self.structure_root_dir = os.path.join(self.structure_root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    # def to_structure(self, img):
    #     # structure_images = []

    #     # for img in imgs:    
    #     img = (img * 255).astype(np.uint8)
    #     img = img[..., ::-1] # RGB -> BGR
    #     if self.structure:
    #         img = cv2.resize(img, (128, 128))
    #     img = self.edge_filter(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    #     # structure_images.append(img_as_float32(img))
    #     return img_as_float32(img)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            # TODO: Use two different images (original: Two frames from the same video)
            # video_ids = np.random.choice(len(self.videos), size=2, replace=False)
            # path = [os.path.join(self.root_dir, self.videos[idx]) for idx in video_ids]
            name = self.videos[idx]
            try:
                
                path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
                if self.structure:
                    structure_path = os.path.join(self.structure_root_dir, os.path.basename(path))
            except ValueError:
                raise ValueError("File formatting is not crrect for id_sampling=True. "
                                "Change file formatting, or set id_sampling=False.")
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            if self.structure:
                structure_path = os.path.join(self.structure_root_dir, name)

        video_name = os.path.basename(path[0])

        if self.is_train and os.path.isdir(path[0]):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32
                

            if self.structure_shape is not None:
                resize_str_fn = partial(resize, output_shape=self.structure_shape)
            else:
                resize_str_fn = img_as_float32

            if type(frames[0]) is bytes:
                video_array = [io.imread(os.path.join(path, frames[idx].decode('utf-8'))) for idx in frame_idx]
                if self.structure:
                    structure_video_array = [io.imread(os.path.join(structure_path, frames[idx].decode('utf-8'))) for idx in frame_idx]
            else:
                video_array = [io.imread(os.path.join(path, frames[idx])) for idx in frame_idx]
                if self.structure:
                    structure_video_array = [io.imread(os.path.join(structure_path, frames[idx])) for idx in frame_idx]

            video_array = [resize_fn(img) for img in video_array]
            if self.structure:
                structure_video_array = [resize_str_fn(img) for img in structure_video_array]

        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            video_array = video_array[frame_idx][..., :3]

            if self.structure:
                structure_video_array = read_video(structure_path, frame_shape = self.structure_shape)
                structure_video_array = structure_video_array[frame_idx][...,:3]

        if self.transform is not None:
            if not self.structure:
                video_array = self.transform(video_array)
            else:
                temp_video_array = video_array + structure_video_array
                temp_video_array = self.transform(temp_video_array)
                video_array = temp_video_array[:2]
                structure_video_array = temp_video_array[2:]

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['driving'] = driving.transpose((2, 0, 1)) # h, w, 3 -> 3, h, w
            out['source'] = source.transpose((2, 0, 1)) # h, w, 3 -> 3, h, w
            if self.structure:
                source_structure = np.array(structure_video_array[0], dtype='float32')
                driving_structure = np.array(structure_video_array[1], dtype='float32')
                out['driving_structure'] = driving_structure.transpose((2, 0, 1)) # h, w, 3 -> 3, h, w
                out['source_structure'] = source_structure.transpose((2, 0, 1)) # h, w, 3 -> 3, h, w
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name
        out['id'] = idx
        
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
