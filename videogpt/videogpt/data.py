import os
import json
import glob
import math
import random
import pickle
import warnings
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
from torchvision.io import read_video
import pytorch_lightning as pl

class CustomJSONDataset(data.Dataset):
    def __init__(self, data_file, sequence_length, train=True, resolution=64):
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.ref_videos = [item['ref_video_path'] for item in self.data]
        self.target_videos = [item['target_video_path'] for item in self.data]
        self.delta_captions = [item['delta_caption'] for item in self.data]

        self.video_clips = VideoClips(self.ref_videos, sequence_length, num_workers=32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ref_video, _, info, video_idx = self.video_clips.get_clip(idx)
        target_video_path = self.target_videos[idx]
        target_video, _, _ = read_video(target_video_path)
        delta_caption = self.delta_captions[idx]

        ref_video = preprocess(ref_video, self.resolution, self.sequence_length)
        target_video = preprocess(target_video, self.resolution, self.sequence_length)

        return {
            'ref_video': ref_video,
            'target_video': target_video,
            'delta_caption': delta_caption
        }

def preprocess(video, resolution, sequence_length=None):
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5
    return video

class VideoData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def _dataset(self, train):
        if self.args.data_format == 'custom_json':
            return CustomJSONDataset(self.args.data_path, self.args.sequence_length, train=train, resolution=self.args.resolution)
        else:
            raise ValueError("Unsupported data format")

    def _dataloader(self, train):
        dataset = self._dataset(train)
        sampler = data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()) if dist.is_initialized() else None
        return data.DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler=sampler, shuffle=sampler is None)

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()
