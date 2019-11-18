# MiCT-Net for Video Classification

This is an implementation of the Mixed Convolutional Tube (MiCT) on PyTorch with a ResNet backbone. The model predicts the human action in each video from a list of 101 classes. It achieves **63.2 top-1 accuracy** on the UCF-101 dataset when tested with video clips of 16 frames. It is based on the work by Y. Zhou, X. Sun, Z-J Zha and W. Zeng described in this [paper](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/Zhou_MiCT_Mixed_3D2D_CVPR_2018_paper.pdf) from Microsoft Research.

This repository includes:

* Source code of MiCT-Net built on the ResNet backbone, and named MiCT-ResNet throughout the rest of this repository

* Source code for 3D-ResNet adapted from [kenshohara](https://github.com/kenshohara/3D-ResNets-PyTorch) and used for performance comparison

* Code to prepare the UCF-101 dataset

* Training and evaluation code for UCF-101

The code is documented and designed to be easy to extend for your own dataset. If you use it in your projects, please consider citing this repository (bibtex below).


## MiCT-ResNet Architecture Overview
This implementation follows the principle of MiCT-Net by introducing a small number of 3D residual convolutions at key locations of a 2D-CNN backbone. The authors observed that 3D ConvNets are limited in depth due their memory requirements and are difficult to train. Their idea is to limit the number of 3D convolution layers while increasing the depth of the feature map using a 2D-CNN.

There are many differences with the paper since the backbones are not the same. The paper uses a custom backbone inspired from Inception. I have chosen to use the ResNet backbone instead to be able to more easily compare my results and to benefit from pre-trained weights on ImageNet.
 
![MiCT-ResNet Architecture](assets/MiCT-ResNet-18-V2.jpg)

As shown above, I have opted for the introduction of five 3D convolutions, one at the entrance of the network and one at the beginning of each of the four main ResNet blocks. After each 3D convolution, we merge features of the two branches with a cross domain element-wise summation. This operation can speed up learning and allow training of deeper architectures. It also allows the 3D convolution branch to only learn residual temporal features, which are the motion of objects and persons in videos, to complement the spatial features learned by 2D convolutions.


## UCF-101 Dataset

[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. It has served the Deep Learning community well for many years and continues to be used. One thing to know when analysing the results below, is that despite its 13320 videos and 100+ clips for each of the 101 action categories, it is actually a relatively small data set for the task because of its low variability. This results in strong over-fitting on most recent architectures which makes their comparison difficult. All the clips are taken from only 2.5k distinct videos. For example there can be 7 clips from one video of a person playing piano. This means that there is far less variation than if the action in each clip was performed by a different person. Example frames of some of the human actions in UCF-101 are shown below.

![UCF101 action examples](assets/UCF101.jpg)


## Training and Test Details

To facilitate results comparison and unless otherwise stated, the procedure described below applies to both MiCT-ResNet and 3D-ResNet. 

* **Backbone:** All experiments are based on the 18 layers version of the ResNet backbone. The temporal stride is 16 and the spatial stride is 32. The first 3D convolution has a temporal stride of 1 to fully harvest the input sequence. Weights are initialised from ImageNet pre-trained weights. For 3D-ResNet, the 3D filters are bootstrapped by repeating the weights of the 2D filters N times along the temporal dimension, and rescaling them by dividing by N. 

* **Optimizer and Regularization:** SGD is used with a learning rate of 1e-2 and a weight decay of 5e-4. These parameters have been selected using grid search. The training of Mict-ResNet-18 lasts 120 epochs with the learning rate divided by 10 after 80 epochs. The training of 3D-ResNet-18 lasts 90 epochs with the learning rate divided by 10 after 40 and 80 epochs. The batch size is set to 128 video clips of 16 frames each. 50% dropout is applied just before the output layer.

* **Data Augmentation:** The literature indicates that UCF-101 does not have enough data and variation to avoid over-fitting on models with 3D convolutions. MiCT-ResNet is no exception. The problem can be resolved by pre-training the networks on Kinetics or any other large scale data set (out of the scope of this repository). I have used the full arsenal of augmentation techniques to reduce over-fitting as much as possible with temporal down-sampling, horizontal flipping, corner/center cropping at different random sizes and aspect ratios (more on this in the next bullet point), plus a combination of random brightness, contrast, and color adjustments. All frames of the same clip are applied identical transformations.

* **Video Preparation:** During training, each video is randomly down-sampled along the temporal dimension and a set of 16 consecutive frames is randomly chosen. The sequence is looped as necessary to obtain 16 frames clip. To support training with a large number of video clips per batch, the model's input size is set to 160x160. All frames are first re-sized to 256x192. The width and height are then independently picked from [128, 144, 160, 176, 192] and a region is cropped either at one of the corners of the image or at its center. The crop is then re-sized to 160x160. The aspect ratio thus randomly varies from 2/3 to 3/2. At test time, the first 16 frames of the video are selected, then resized to 256x192 and a center crop of 160x160 is extracted for all frames.

* **Hardware:** All experiments were done on a single Titan RTX with 24GB of RAM. For smaller configurations, consider reducing the input frames to 112x112 and/or applying a temporal stride of 2 on the first 3D convolution instead of the max pooling layer.


## Results

This section reports test results for the following experiments:
 * MiCT-ResNet-18 versus 3D-ResNet-18
 * MiCT-ResNet-18 with varying kernel sizes for the first 3D convolution: 3x7x7x, 5x7x7, and 7x7x7
 
The models are evaluated against the **top1** and **top5** accuracies. All results are averaged across the 3 standard splits. **MiCT-ResNet-18 leads by 1.4 point while being 3.1 times faster** which confirms the validity of the approach of the authors. The memory size is given for the processing of one video clip of 16 frames at a time (ie. batch size of one).

|                                                         | Parameters  | Top1 / Top5     | Memory size |     FPS     |
|---------------------------------------------------------|-------------|-----------------|-------------|-------------|
| [MiCT-ResNet-18](results/MiCT-ResNet-18-7x7x7.jpg)      | 16.1M       | **63.2** / 83.7 | 985 MB      | **1981**    |
| [3D-ResNet-18](results/3D-ResNet-18.jpg)                | 33.3M       | 61.8 / 83.3     | 1045 MB     | 644         |

As shown below, the size of the first 3D kernel has a significant impact on the efficiency of the MiCT-ResNet architecture. Harvesting 7 consecutive RGB input frames provides the best accuracy but impacts inference speed.

| First 3D kernel size    | Parameters  | Top1 / Top5     | Memory size |     FPS     |
|-------------------------|-------------|-----------------|-------------|-------------|
| 3x7x7                   | 16.01M      | 61.4 / 83.3     | 983 MB      | 2380        |
| 5x7x7                   | 16.03M      | 62.7 / 83.4     | 985 MB      | 2147        |
| 7x7x7                   | 16.05M      | **63.2** / 83.7 | 985 MB      | 1981        |

It remains to be seen how MiCT-ResNet-18 and 3D-ResNet-18 compare if they were both pre-trained on ImageNet & Kinetics. Let me know if you have access to the Kinetics data set and are willing to contribute!


## Training on Your Own
You can train, evaluate and predict directly from the command line as such:

```
# Training a new MiCT-ResNet-18 model starting from pre-trained ImageNet weights
python train.py --model mictresnet --lr 1e-2 --weight-decay 5e-4 --batch-size 128 
                --base-size 192 --crop-size 160 --split 1 --checkname mictresnet_v1 
                --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80
```

You can also evaluate the model with:
```
# Evaluate 3D-ResNet-18 model on UCF-101 split 1 test set
python test.py --model 3dresnet --test-batch-size 1 --base-size 192 --crop-size 160 
               --resume /path/to/your/checkpoint/tar --split 1 --crop-vid 16
```

You can also test the inference speed on your hardware using this command:
```
# Test the inference speed
python test_speed.py --model mictresnet
```

To prepare the dataset once download is complete run:

```
# Extract all frames from all videos and create the train/test lists for the 3 splits.
python prepare_ucf101.py --download-dir /path/to/your/downloaded/tar
```


## Requirements
Python 3.7, Torch 1.0 or greater, and tqdm.


## Citation
Use this bibtex to cite this repository:
```
@misc{fmahoudeau_mict_net_2019,
  title={Mixed Convolutional Tube (MiCT-Net) for video classification in PyTorch},
  author={Florent Mahoudeau},
  year={2019},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/fmahoudeau/MiCT-Net-PyTorch}},
}
