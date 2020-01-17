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

import time
import numpy as np
import torch
from option import Options

import sys
sys.path.insert(0, '../../')

from mictnet.models import get_classification_model


def test_speed(args):
    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    print('Compute device: ' + device)
    device = torch.device(device)

    # model
    model_kwargs = {'backbone': args.backbone, 'version': args.version} \
        if args.model == 'mictresnet' else {}
    model = get_classification_model(args.model, **model_kwargs)

    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)

    model.to(device)
    model.eval()

    run_time = list()

    for i in range(0, 1000):
        input = torch.randn(1, 3, 16, 160, 160).to(device)
        # ensure that context initialization and normal_() operations
        # finish before you start measuring time
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input)

        torch.cuda.synchronize()  # wait for mm to finish
        end = time.perf_counter()
        run_time.append(end-start)

    run_time.pop(0)

    print('Mean running time is {:.5f}'.format(np.mean(run_time)))
    print('FPS = {:.1f}'.format(16 / np.mean(run_time)))


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    test_speed(args)
