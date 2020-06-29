#!/bin/sh
pip install -r requirements.txt &&
# pip install --upgrade torch torchvision   &&
cd ~/../yy-volume/codes/multi_task &&
python train.py --dataroot ../../datasets/mr_ct_raw_dataset/pet/  --name pet --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/pet --display_id 0 --gpu_ids 0  --log_dir logs/pet/ --save_epoch_freq 10 --arch resnet18  --batch_size 3  --rgr_lr 0.001 --fc_input 512 
