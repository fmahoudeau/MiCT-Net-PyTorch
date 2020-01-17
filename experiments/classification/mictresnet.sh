#!/usr/bin/env bash

CHECKNAME=mictresnet18_v1_drop60_split
DROPOUT=0.6

python train.py --model mictresnet --version v1 --backbone resnet18 --lr 1e-2 --weight-decay 5e-4 --dropout $DROPOUT --batch-size 112 --base-size 192 --crop-size 160 --checkname "$CHECKNAME"1 --split 1 --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80
python train.py --model mictresnet --version v1 --backbone resnet18 --lr 1e-2 --weight-decay 5e-4 --dropout $DROPOUT --batch-size 112 --base-size 192 --crop-size 160 --checkname "$CHECKNAME"2 --split 2 --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80
python train.py --model mictresnet --version v1 --backbone resnet18 --lr 1e-2 --weight-decay 5e-4 --dropout $DROPOUT --batch-size 112 --base-size 192 --crop-size 160 --checkname "$CHECKNAME"3 --split 3 --crop-vid 16 --epochs 120 --pretrained --lr-scheduler step --lr-step 80

python test.py --model mictresnet --version v1 --backbone resnet18 --test-batch-size 1 --base-size 192 --crop-size 160 --checkname "$CHECKNAME"1 --resume output/ucf101/mictresnet/"$CHECKNAME"1/model_best.pth.tar --split 1 --crop-vid 16
python test.py --model mictresnet --version v1 --backbone resnet18 --test-batch-size 1 --base-size 192 --crop-size 160 --checkname "$CHECKNAME"2 --resume output/ucf101/mictresnet/"$CHECKNAME"2/model_best.pth.tar --split 2 --crop-vid 16
python test.py --model mictresnet --version v1 --backbone resnet18 --test-batch-size 1 --base-size 192 --crop-size 160 --checkname "$CHECKNAME"3 --resume output/ucf101/mictresnet/"$CHECKNAME"3/model_best.pth.tar --split 3 --crop-vid 16
