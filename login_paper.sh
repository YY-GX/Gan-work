#!/bin/sh
pip install -r requirements.txt &&
pip install --upgrade torch torchvision   &&
cd ~/../yy-volume/codes/multi_task &&
python train.py --dataroot ../../datasets/mr_ct_raw_dataset/  --name full_0_pretrain --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/full_0_pretrain_slow --display_id 0 --gpu_ids 0,1,2,3  --log_dir logs/full_0_pretrain_slow/ --save_epoch_freq 5 --arch resnet18  --batch_size 12  --rgr_lr 0.001 --fc_input 512 --print_freq 100

