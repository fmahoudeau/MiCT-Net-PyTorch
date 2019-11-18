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

import matplotlib.pyplot as plt


# Learning curve
train_loss, train_top1 = [], []
val_loss, val_top1 = [], []
with open('/home/fanos/PycharmProjects/MiCTNet/3dresnet.txt', 'r') as f:
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
acc=np.array([[63.9,63.3,60.5],
              [65.5,65.5,62],
              [67.5,67,64],
              [68.6,68.2,65.3],
              [68.9,69.2,65.7]]).mean(axis=1)
plt.figure(); plt.plot([16, 35, 75, 150, 300], acc);
plt.xticks([16, 35, 75, 150, 300], ['16', '35', '75', '150', '300'])
plt.xlabel('Number of video clip frames')
plt.ylabel('Top-1 accuracy')
plt.title('MiCT-ResNet-18')
