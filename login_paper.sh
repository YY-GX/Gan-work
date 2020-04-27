#!/bin/sh
pip install -r requirements.txt &&
pip install --upgrade torch torchvision   &&
cd ~/../yy-volume/codes &&
python main_times.py -a resnet50 --logpath logs/log_sepe_comp_50/ --resumedir checkpoints/checkpoints_sepe_comp_50/  --traindatapath ../datasets/mr_ct_raw_dataset/trainB/ --testdatapath ../datasets/mr_ct_raw_dataset/testB/ --augement toge --augedir ../datasets/mr_ct_trans_dataset/50_fake_ct/ --filename toge.csv --epochs 100 --times 20 --dropout 0.5 -b 64 datasets