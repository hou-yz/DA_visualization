#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_DA.py -d digits --source svhn --target mnist &
CUDA_VISIBLE_DEVICES=1 python train_DA.py -d digits --source usps --target mnist &
CUDA_VISIBLE_DEVICES=2 python train_DA.py -d digits --source mnist --target usps &
wait
#CUDA_VISIBLE_DEVICES=0 python train_DA.py -d office31 --source amazon --target webcam &
#CUDA_VISIBLE_DEVICES=1 python train_DA.py -d office31 --source dslr --target webcam &
#CUDA_VISIBLE_DEVICES=2 python train_DA.py -d office31 --source webcam --target dslr &
#wait
#CUDA_VISIBLE_DEVICES=0 python train_DA.py -d office31 --source amazon --target dslr &
#CUDA_VISIBLE_DEVICES=1 python train_DA.py -d office31 --source dslr --target amazon &
#CUDA_VISIBLE_DEVICES=2 python train_DA.py -d office31 --source webcam --target amazon &
#wait
#CUDA_VISIBLE_DEVICES=0 python train_DA.py -d visda &
#CUDA_VISIBLE_DEVICES=1 python train_DA.py -d visda --da_setting mmd &
#CUDA_VISIBLE_DEVICES=2 python train_DA.py -d visda --da_setting adda &
#wait


CUDA_VISIBLE_DEVICES=0 python train_SFIT.py -d digits --source svhn --target mnist &
CUDA_VISIBLE_DEVICES=1 python train_SFIT.py -d digits --source usps --target mnist &
CUDA_VISIBLE_DEVICES=2 python train_SFIT.py -d digits --source mnist --target usps &
wait
#CUDA_VISIBLE_DEVICES=0 python train_SFIT.py -d office31 --source amazon --target webcam &
#CUDA_VISIBLE_DEVICES=1 python train_SFIT.py -d office31 --source dslr --target webcam &
#CUDA_VISIBLE_DEVICES=2 python train_SFIT.py -d office31 --source webcam --target dslr &
#wait
#CUDA_VISIBLE_DEVICES=0 python train_SFIT.py -d office31 --source amazon --target dslr &
#CUDA_VISIBLE_DEVICES=1 python train_SFIT.py -d office31 --source dslr --target amazon &
#CUDA_VISIBLE_DEVICES=2 python train_SFIT.py -d office31 --source webcam --target amazon &
#wait
#CUDA_VISIBLE_DEVICES=0 python train_SFIT.py -d visda &
#CUDA_VISIBLE_DEVICES=1 python train_SFIT.py -d visda --da_setting mmd &
#CUDA_VISIBLE_DEVICES=2 python train_SFIT.py -d visda --da_setting adda &
#wait