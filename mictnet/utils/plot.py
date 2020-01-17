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

import numpy as np
import matplotlib.pyplot as plt


# Learning curve
train_loss, train_top1 = [], []
val_loss, val_top1 = [], []
with open('/path/to/your/training/logs/file.log', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        if line.startswith('train'):
            tokens = line.split(' ')
            train_loss.append(float(tokens[1][:-1]))
            train_top1.append(float(tokens[3][:-1]))
        elif line.startswith('val'):
            tokens = line.split(' ')
            val_loss.append(float(tokens[3][:-1]))
            val_top1.append(float(tokens[5][:-1]))

plt.clf()
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.plot(train_loss, 'b', label="Train loss")
ax1.plot(val_loss, 'b--', label="Validation loss")
ax2.plot(train_top1, 'r', label='Train top1')
ax2.plot(val_top1, 'r--', label='Validation top1')

plt.title('Learning curve for 3D-ResNet-18')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Top1 Accuracy')
ax1.legend(loc="lower center")
ax2.legend(loc="center")
plt.savefig('3D-ResNet-18.jpg');

# Top-1 accuracy as a function of input sequence length
acc_resnet18 = np.array([
    [67.1, 65.5, 64.7],
    [68.1, 66.7, 66.1],
    [69.2, 68.5, 67.3],
    [69.2, 69.3, 67.9],
    [69.6, 70.0, 68.3]]).mean(axis=1)

acc_resnet34 = np.array([
    [69.0, 68.8, 67.7],
    [70.3, 70.0, 69.0],
    [72.1, 71.1, 70.8],
    [73.5, 71.9, 72.0],
    [73.8, 72.5, 72.0]]).mean(axis=1)

plt.figure()
plt.plot([16, 35, 75, 150, 300], acc_resnet18, label='MiCT-ResNet18')
plt.plot([16, 35, 75, 150, 300], acc_resnet34, label='MiCT-ResNet34')
plt.xticks([16, 35, 75, 150, 300], ['16', '35', '75', '150', '300'])
plt.xlabel('Number of video frames')
plt.ylabel('Averaged Top-1 accuracy')
plt.legend(loc='lower right')
plt.title('MiCT-ResNet')
