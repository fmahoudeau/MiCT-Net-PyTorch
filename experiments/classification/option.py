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
import argparse
import torch


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Video Classification')

        # model and dataset
        parser.add_argument('--model', type=str, default='mictresnet',
                            help='model name (default: mictresnet)')
        parser.add_argument('--version', type=str, default='v1',
                            help='model variant (default: v1)')
        parser.add_argument('--backbone', type=str, default='resnet18',
                            help='backbone name (default: resnet18)')
        parser.add_argument('--dataset', type=str, default='ucf101',
                            help='dataset name (default: ucf101)')
        parser.add_argument('--split', type=int, default=1,
                            help='dataset split number (default: 1)')
        parser.add_argument('--data-folder', type=str,
                            default=os.path.join(os.environ['HOME'],
                                                 'Documents/Datasets/ucf101'),
                            help=' dataset folder (default: \
                            $(HOME)/Documents/Datasets/ucf101)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=256,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        parser.add_argument('--crop-vid', type=int, default=16,
                            help='video frames crop length')
        # training hyper params
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start-epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=128,
                            metavar='N', help='input batch size for \
                            training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout during training (default: 0.5)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--lr-step', type=int, default=50,
                            help='steps to decay learning rate by 0.1')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--gpu-id', type=int, default=0,
                            help='gpu device id (default: 0)')
        # checking point
        parser.add_argument('--resume', type=str, default='',
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='MictResNet',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--pretrained', action='store_true', default=False,
                            help='load weights pretrained on imagenet')
        parser.add_argument('--ft', action='store_true', default=False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating top1 and top5 accuracies')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test folder')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, test_batch_size and lr
        if args.epochs is None:
            epoches = {
                'ucf101': 120,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'ucf101': 0.01,
            }
            args.lr = lrs[args.dataset.lower()]
        return args
