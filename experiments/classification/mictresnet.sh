#!/usr/bin/env bash

python train.py --model mictresnet --lr 1e-2 --weight-decay 5e-4 --batch-size 128 --base-size 192 --crop-size 160 --checkname mictresnet_7x7x7_split1 --split 1 --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80
python train.py --model mictresnet --lr 1e-2 --weight-decay 5e-4 --batch-size 128 --base-size 192 --crop-size 160 --checkname mictresnet_7x7x7_split2 --split 2 --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80
python train.py --model mictresnet --lr 1e-2 --weight-decay 5e-4 --batch-size 128 --base-size 192 --crop-size 160 --checkname mictresnet_7x7x7_split3 --split 3 --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80

python test.py --model mictresnet --test-batch-size 1 --base-size 192 --crop-size 160 --resume runs/ucf101/mictresnet/mictresnet_7x7x7_split1/model_best.pth.tar --split 1 --crop-vid 16
python test.py --model mictresnet --test-batch-size 1 --base-size 192 --crop-size 160 --resume runs/ucf101/mictresnet/mictresnet_7x7x7_split2/model_best.pth.tar --split 2 --crop-vid 16
python test.py --model mictresnet --test-batch-size 1 --base-size 192 --crop-size 160 --resume runs/ucf101/mictresnet/mictresnet_7x7x7_split3/model_best.pth.tar --split 3 --crop-vid 16
