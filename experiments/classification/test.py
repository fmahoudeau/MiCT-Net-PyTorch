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
import logging
from tqdm import tqdm
import torch
from torch.utils import data
import torchvision.transforms as transform

import sys
sys.path.insert(0, '../../')

from option import Options
from mictnet.models import get_classification_model
from mictnet.datasets import get_classification_dataset
from mictnet import utils


def test(args):
    logger, console, output_dir = utils.file.create_logger(args, 'val')

    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    logger.info('Compute device: ' + device)
    device = torch.device(device)

    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    # dataset
    data_kwargs = {'logger': logger, 'transform': input_transform,
                   'base_size': args.base_size, 'crop_size': args.crop_size,
                   'crop_vid': args.crop_vid, 'split': args.split,
                   'root': args.data_folder}
    testset = get_classification_dataset(args.dataset, mode='val', **data_kwargs)

    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    testloader = data.DataLoader(testset, batch_size=args.test_batch_size,
                                 drop_last=False, shuffle=False, **loader_kwargs)

    # model
    model_kwargs = {'backbone': args.backbone, 'version': args.version} \
        if args.model == 'mictresnet' else {}
    model = get_classification_model(args.model, **model_kwargs)

    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume,
                                                              checkpoint['epoch']))

    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total number of parameters: %d" % pytorch_total_params)

    # don't output to stdout anymore when logging
    logging.getLogger('').removeHandler(console)

    # validation
    model.to(device)
    model.eval()
    top1 = utils.AverageMeter('acc@1', ':6.2f')
    top5 = utils.AverageMeter('acc@5', ':6.2f')

    tbar = tqdm(testloader, desc='\r')
    for i, (video, target) in enumerate(tbar):
        video = video.to(device)
        target = target.to(device)
        with torch.no_grad():
            pred = model(video)
            acc1, acc5 = utils.accuracy(pred, target, topk=(1, 5))
            top1.update(acc1[0], args.batch_size)
            top5.update(acc5[0], args.batch_size)
        tbar.set_description(
            'acc1: %.3f, acc5: %.3f' % (top1.avg, top5.avg))

    logger.info('acc1: %.3f, acc5: %.3f' % (top1.avg, top5.avg))


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    test(args)
