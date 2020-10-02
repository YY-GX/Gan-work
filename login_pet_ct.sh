#!/bin/sh
pip install -r requirements.txt
pip install --upgrade torch torchvision
cd ~/../yy-volume/codes/
python main_times.py -a resnet50 --logpath logs/log_pet_ct/ --resumedir checkpoints/checkpoints_pet_ct/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainB/ --testdatapath ../datasets/final_mr_ct/valB/ --augement toge  --epochs 100 --times 1 --augedir  ../datasets/final_mr_ct/pet_fake -b 64 --dropout 0.5  datasets

