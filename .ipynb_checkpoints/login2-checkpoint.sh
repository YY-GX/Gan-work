#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 
cd ~/../yy-volume/codes/ 
python main_times.py -a resnet50 --logpath logs/log_sepe_comp_200/ --resumedir checkpoints/checkpoints_seperate_comp_200/  --traindatapath ../datasets/mr_ct_raw_dataset/trainB/ --testdatapath ../datasets/mr_ct_raw_dataset/testB/ --augement toge --augedir ../datasets/mr_ct_trans_dataset/full_0/ --filename 200_200.csv --epochs 100 --times 20 --dropout 0.5 -b 64 datasets --gpu 0 
  
# cd ~/../yy-volume/codes/
# python main_covid.py --epochs 300 -j 8 --logpath logs/log_covid_2/ --resumedir checkpoints/checkpoints_covid_2/ -a resnet50 -b 64 --wd 1e-4 ../datasets/x2ct/covid_ct_v2/
# python ~/../yy-volume/x2ct/x2ct/main.py -b 32
# python net_train.py
# python ../x2ct/moco/main_moco.py \
#   -a resnet50 \
#   --lr 0.0075 \
#   --batch-size 64 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   '../datasets/x2ct/covid_ct_split_official'

