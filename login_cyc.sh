#!/bin/sh
pip install -r requirements.txt &&
pip install --upgrade torch torchvision   &&
cd ~/../yy-volume/codes/multi_task &&
python train.py --dataroot ../../datasets/mr_ct_raw_dataset/  --name full_0_pretrain --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/full_0_pretrain --display_id 0 --gpu_ids 0,1,2,3  --log_dir logs/full_0_pretrain/ --save_epoch_freq 5 --arch resnet18  --batch_size 16  --rgr_lr 0.001 --fc_input 512 --print_freq 100


# python main_covid.py --epochs 300 -j 8 --logpath logs/log_covid_2/ --resumedir checkpoints/checkpoints_covid_2/ -a resnet50 -b 64 --wd 1e-4 ../datasets/x2ct/covid_ct_v2/
# python ~/../yy-volume/x2ct/x2ct/main.py -b 16
# tensorboard --logdir=./../x2ct/DANN_py3/logs --port 6006

# cd ~/../yy-volume/x2ct/imbalanced-dataset-sampler &&
# python setup.py install &&
 
# cd ~/../yy-volume/x2ct/DANN_py3/logs/ &&
# rm -r * &&



# cd ~/../yy-volume/codes &&

# CUDA_VISIBLE_DEVICES=0,1  
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-4 -wd 1e-4 -num 0 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-4 -wd 1e-3 -num 1 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-4 -wd 1e-2 -num 2 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-4 -wd 1e-1 -num 3 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-4 -num 4
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-3 -num 5
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-2 -num 6
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-1 -num 7
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-5 -wd 1e-4 -num 8
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-5 -wd 1e-3 -num 9
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-5 -wd 1e-2 -num 10
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 5e-5 -wd 1e-1 -num 11
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-4 -num 12
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-3 -num 13
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-2 -num 14 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-3 -num 15
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-7 -wd 1e-2 -num 5
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-8 -wd 1e-2 -num 13 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-1 -num 0
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-1 -num 1
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-1 -num 2
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-1 -num 3
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-2 -num 4
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-3 -num 5
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-4 -num 6
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-7 -wd 1e-3 -num 7
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-4 -num 8
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-3 -num 9
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-2 -num 10
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-5 -wd 1e-1 -num 11

# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-4 -wd 1e-4 -num 12
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-4 -wd 1e-3 -num 13
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-4 -wd 1e-2 -num 14 
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-4 -wd 1e-1 -num 15


# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-4 -num 4

# python net_train.py

 
