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

"""Prepare UCF101 dataset."""
import os
import argparse
import glob
from subprocess import call


_TARGET_DIR = os.path.join(os.environ['HOME'], 'Documents/Datasets/ucf101')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download and prepare UCF101 dataset.',
        epilog='Example: python prepare_ufc101.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=_TARGET_DIR,
                        help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true',
                        help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite downloaded files if set, in case they are corrupted')
    args = parser.parse_args()
    return args


def download_ufc101(data_dir, overwrite=False):
    raise NotImplementedError


def extract_ufc101_frames(data_dir, overwrite=False):
    if not os.path.exists(os.path.join(data_dir, 'images')):
        os.makedirs(os.path.join(data_dir, 'images'))

    for filepath in glob.glob(os.path.join(data_dir, 'videos/*/*.avi')):
        splitted_path = filepath.split(os.path.sep)
        category = splitted_path[-2]
        filename = splitted_path[-1]
        dest = os.path.join(data_dir, 'images', category,
                            filename[:-4] + '-%04d.jpg')
        if not os.path.exists(os.path.join(data_dir, 'images', category)):
            os.makedirs(os.path.join(data_dir, 'images', category))

        print('Extracting ' + category + '/' + filename)
        call(["ffmpeg", "-i", filepath, dest])


def train_test_split(data_dir, split='01'):
    categories_file = os.path.join(data_dir, 'UCF101TrainTestSplits-RecognitionTask', 'classInd.txt')
    train_file = os.path.join(data_dir, 'UCF101TrainTestSplits-RecognitionTask', 'trainlist' + split + '.txt')
    test_file = os.path.join(data_dir, 'UCF101TrainTestSplits-RecognitionTask', 'testlist' + split + '.txt')

    with open(categories_file, 'r') as f:
        categories = dict([line.rstrip('\n').split(' ')[::-1] for line in f.readlines()])

    # Build the train list.
    with open(train_file, 'r') as f:
        train_list = [line.rstrip('\n') for line in f.readlines()]
        train_list = [line.split(' ')[0] for line in train_list]
        train_list = [line[:-4] for line in train_list]

    # Build the test list.
    with open(test_file, 'r') as f:
        test_list = [line.rstrip('\n')[:-4] for line in f.readlines()]

    # Set the groups in a dictionary.
    for group_name, filenames in {'train': train_list, 'val': test_list}.items():
        print('Preparing %s.csv' % (group_name+str(int(split))))
        n_videos, n_frames = 0, 0
        with open(os.path.join(data_dir, 'images', group_name+str(int(split))+'.csv'), 'w') as f:
            for filename in filenames:
                category, basename = filename.split('/')
                n_files = len(glob.glob(os.path.join(data_dir, 'images', category, basename + '*.jpg')))
                f.write(filename + ',' + str(n_files) + ',' + categories[category] + '\n')
                n_videos += 1
                n_frames += n_files

        print('%d videos for a total of %d frames ' % (n_videos, n_frames))


if __name__ == '__main__':
    args = parse_args()
    extract_ufc101_frames(args.download_dir)
    for split in ['01', '02', '03']:
        train_test_split(args.download_dir, split)
