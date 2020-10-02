#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 
cd ~/../yy-volume/codes/ 
python main_times.py -a resnet50 --logpath logs/log_baseline_tl/ --resumedir checkpoints/checkpoints_baseline_tl/ --filename pure.csv --traindatapath ../datasets/mr_ct_raw_dataset/trainA/ --testdatapath ../datasets/mr_ct_raw_dataset/testA/ --augement mr_pure  --epochs 100 --times 1  -b 64 --dropout 0.5 --lr 1e-4  datasets 