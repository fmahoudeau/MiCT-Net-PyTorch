#!/usr/bin/env bash

python train.py --model resnet3d --lr 1e-2 --weight-decay 5e-4 --batch-size 128 --base-size 192 --crop-size 160 --checkname resnet3d_split1 --split 1 --crop-vid 16 --epochs 90 --pretrained --lr-scheduler step --lr-step 40
python train.py --model resnet3d --lr 1e-2 --weight-decay 5e-4 --batch-size 128 --base-size 192 --crop-size 160 --checkname resnet3d_split2 --split 2 --crop-vid 16 --epochs 90 --pretrained --lr-scheduler step --lr-step 40
python train.py --model resnet3d --lr 1e-2 --weight-decay 5e-4 --batch-size 128 --base-size 192 --crop-size 160 --checkname resnet3d_split3 --split 3 --crop-vid 16 --epochs 90 --pretrained --lr-scheduler step --lr-step 40

python test.py --model resnet3d --test-batch-size 1 --base-size 192 --crop-size 160 --resume runs/ucf101/resnet3d/resnet3d_split1/model_best.pth.tar --split 1 --crop-vid 16
python test.py --model resnet3d --test-batch-size 1 --base-size 192 --crop-size 160 --resume runs/ucf101/resnet3d/resnet3d_split2/model_best.pth.tar --split 2 --crop-vid 16
python test.py --model resnet3d --test-batch-size 1 --base-size 192 --crop-size 160 --resume runs/ucf101/resnet3d/resnet3d_split3/model_best.pth.tar --split 3 --crop-vid 16
