# ==============================================================================
# Copyright 2019 Florent Mahoudeau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import io
import random
import glob
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor


__all__ = ['UCF101Classification']

"""Loads the UCF101 dataset and prepares it for train, val and test phases."""


class UCF101Classification:
    CLASSES = [
        'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
        'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
        'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
        'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
        'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
        'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
        'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering',
        'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
        'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
        'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting',
        'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
        'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf',
        'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
        'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
        'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
        'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
        'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
        'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
        'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
        'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
    ]
    N_CLASSES = 101

    def __init__(self, logger, root, mode=None, transform=None,
                 base_size=256, crop_size=224, crop_vid=16,
                 split=1, pre_load=False, verify=False, **kwargs):
        self.totensor = ToTensor()
        self.mode = mode
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.crop_vid = crop_vid
        self.split = split
        self.pre_load = pre_load

        image_dir = os.path.join(root, 'images')
        splits_dir = image_dir  # train/val splits are pre-cut
        if self.mode == 'train':
            logger.info('Loading UCF101 ' + ('train%d.csv' % self.split))
            split_f = os.path.join(splits_dir, 'train%d.csv' % self.split)
        elif self.mode == 'val':
            logger.info('Loading UCF101 ' + ('val%d.csv' % self.split))
            split_f = os.path.join(splits_dir, 'val%d.csv' % self.split)
        else:
            raise RuntimeError('Unknown dataset split')

        self.videos = []
        self.labels = []
        with open(os.path.join(split_f), "r") as lines:
            for line in tqdm(lines):
                video_id, n_frames, category_id = line.rstrip('\n').split(',')
                n_frames = int(n_frames)
                image_base = os.path.join(image_dir, video_id)
                # Verifying all images is time consuming.
                if verify:
                    n_files = len(glob.glob(image_base + '*.jpg'))
                    assert n_frames == n_files, '{}: {} <> {}'.format(video_id, n_frames, n_files)
                # Preload raw bytes from disk.
                if self.pre_load:
                    vid_bytes = []
                    for i in range(n_frames):
                        vid_bytes.append(bytearray(open(image_base + '-{:04}.jpg'.format(i+1), 'rb').read()))
                    self.videos.append(vid_bytes)
                # Prepare list of image file names.
                else:
                    filenames = []
                    for i in range(n_frames):
                        filenames.append(image_base + '-{:04}.jpg'.format(i + 1))
                    self.videos.append(filenames)
                self.labels.append(int(category_id)-1)

        assert (len(self.videos) == len(self.labels))

    def __getitem__(self, index):
        vid = self._load_sample(self.videos[index])
        target = self.labels[index]

        # synchronized transform
        if self.mode == 'train':
            vid = self._train_transform(vid)
        else:
            vid = self._val_transform(vid)

        # general resize, normalize and to tensor
        if self.transform is not None:
            vid = [self.transform(vid[i]) for i in range(len(vid))]
        else:
            vid = [self.totensor(vid[i]) for i in range(len(vid))]

        # final transform
        return torch.stack(vid, dim=1), target

    def __len__(self):
        return len(self.videos)

    @property
    def n_classes(self):
        return self.N_CLASSES

    def _load_sample(self, images):
        if self.mode == 'train':
            # random down-sampling
            images = images[::random.randint(1, 4)]
            # pick a random sequence
            idx = random.randint(0, max(0, len(images)-self.crop_vid))
            # keep a maximum of crop_vid frames (25 fps)
            images = images[idx:idx+self.crop_vid]
        else:
            # no down-sampling
            # keep a maximum of crop_vid frames (25 fps)
            images = images[:self.crop_vid]

        # load remaining frames
        vid = []
        if self.pre_load:
            for im_bytes in images:
                vid.append(Image.open(io.BytesIO(im_bytes)).convert('RGB'))
        else:
            for im_filename in images:
                vid.append(Image.open(im_filename).convert('RGB'))

        # loop the indices as many times as necessary to satisfy the length
        for image in vid:
            if len(vid) >= self.crop_vid:
                break
            vid.append(image)

        return vid

    def _val_transform(self, vid):
        # rescale
        short_size = self.base_size
        w, h = vid[0].size
        oh = short_size
        ow = int(1.0 * w * oh / h)
        vid = [vid[i].resize((ow, oh), Image.BILINEAR) for i in range(len(vid))]

        # center crop
        outsize = self.crop_size
        w, h = vid[0].size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        vid = [vid[i].crop((x1, y1, x1+outsize, y1+outsize)) for i in range(len(vid))]

        return vid

    def _train_transform(self, vid):
        # random mirror
        if random.random() < 0.5:
            vid = [vid[i].transpose(Image.FLIP_LEFT_RIGHT) for i in range(len(vid))]

        # rescale
        w, h = vid[0].size
        oh = self.base_size
        ow = int(1.0 * w * oh / h)
        vid = [vid[i].resize((ow, oh), Image.BILINEAR) for i in range(len(vid))]

        # pad to crop_size
        if self.base_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            vid = [ImageOps.expand(vid[i], border=(0, 0, padw, padh), fill=0) for i in range(len(vid))]

        # random size crops at the corners or center
        vid = self._multi_scale_corner_crop(vid)

        # random color, brightness, and contrast adjustments
        color = np.random.uniform(0.5, 1.5)
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.5, 1.5)
        vid = [ImageEnhance.Color(vid[i]).enhance(color) for i in range(len(vid))]
        vid = [ImageEnhance.Brightness(vid[i]).enhance(brightness) for i in range(len(vid))]
        vid = [ImageEnhance.Contrast(vid[i]).enhance(contrast) for i in range(len(vid))]

        return vid

    def _random_crop(self, vid):
        # random crop to crop_size
        w, h = vid[0].size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        vid = [vid[i].crop((x1, y1, x1+self.crop_size, y1+self.crop_size)) for i in range(len(vid))]
        return vid

    def _multi_scale_corner_crop(self, vid, scales=[128, 144, 160, 176, 192]):
        w, h = vid[0].size
        crop_width = scales[random.randint(0, len(scales) - 1)]
        crop_height = scales[random.randint(0, len(scales) - 1)]
        crop_position = ['c', 'tl', 'tr', 'bl', 'br'][random.randint(0, 4)]

        if crop_position == 'c':
            center_x = w // 2
            center_y = h // 2
            x1 = center_x - crop_width // 2
            y1 = center_y - crop_height // 2
            x2 = center_x + crop_width // 2
            y2 = center_y + crop_height // 2
        elif crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_width
            y2 = crop_height
        elif crop_position == 'tr':
            x1 = w - crop_width
            y1 = 0
            x2 = w
            y2 = crop_height
        elif crop_position == 'bl':
            x1 = 0
            y1 = h - crop_height
            x2 = crop_width
            y2 = h
        elif crop_position == 'br':
            x1 = w - crop_width
            y1 = h - crop_height
            x2 = w
            y2 = h

        vid = [vid[i].crop((x1, y1, x2, y2)) for i in range(len(vid))]
        vid = [vid[i].resize((self.crop_size, self.crop_size), Image.BILINEAR) for i in range(len(vid))]
        return vid
