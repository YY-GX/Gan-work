#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 
cd ~/../yy-volume/codes/ 
python main_times.py -a resnet50 --logpath logs/log_pure/ --resumedir checkpoints/checkpoints_pure/ --filename pure.csv --traindatapath ../datasets/mr_ct_raw_dataset/trainB/ --testdatapath ../datasets/mr_ct_raw_dataset/testB/ --augement pure  --epochs 100 --times 30  -b 64 --dropout 0.5  datasets

# python main_times.py -a resnet50 --logpath logs/log_sepe_comp_100/ --resumedir checkpoints/checkpoints_seperate_comp_100/  --traindatapath ../datasets/mr_ct_raw_dataset/trainB/ --testdatapath ../datasets/mr_ct_raw_dataset/testB/ --augement toge --augedir ../datasets/mr_ct_trans_dataset/100_100_fake_ct/ --filename toge.csv --epochs 100 --times 10 --dropout 0.5 -b 64 datasets 

# python main_times.py -a resnet50 --logpath logs/log_sepe_comp/ --resumedir checkpoints/checkpoints_seperate_comp/  --traindatapath ../datasets/mr_ct_raw_dataset/trainB/ --testdatapath ../datasets/mr_ct_raw_dataset/testB/ --augement toge --augedir ../datasets/mr_ct_trans_dataset/150_50_fake_ct/ --filename 150_50.csv --epochs 100 --times 10 --dropout 0.5 -b 64 datasets

# python main_times.py -a resnet50 --logpath logs/log_toge/ --resumedir checkpoints/checkpoints_toge/  --traindatapath ../datasets/mr_ct_raw_dataset/trainB/ --testdatapath ../datasets/mr_ct_raw_dataset/testB/ --augement toge --augedir ../datasets/mr_ct_trans_dataset/fake_ct_train/ --filename toge.csv --epochs 100 --times 10 --dropout 0.5 -b 64 datasets

# cd ~/../yy-volume/codes/
# python main_covid.py --epochs 300 -j 8 --logpath logs/log_covid_2/ --resumedir checkpoints/checkpoints_covid_2/ -a resnet50 -b 64 --wd 1e-4 ../datasets/x2ct/covid_ct_v2/
# python ~/../yy-volume/x2ct/x2ct/covid_train.py --epochs 500 -j 8 --logpath logs/log_covid_2/ --resumedir data -a resnet50 -b 64 --wd 1e-4 ../datasets/x2ct/covid_ct_v2/
# python ~/../yy-volume/x2ct/x2ct/covid_train.py data 
# python net_train.py

# cd ~/../yy-volume/x2ct/DANN_py3/logs1/
# rm -r *
# cd ~/../yy-volume/codes
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-3 -num 0 
# python ~/../yy-volume/x2ct/DANN_py3/covid_train.py data 
# python ../x2ct/moco/main_moco.py \
#   -a resnet50 \
#   --lr 0.015 \
#   --batch-size 128 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   '../datasets/x2ct/covid_ct_split_official'

# cd ~/../yy-volume/x2ct/DANN_py3_v2/logs/
# rm -r *

# cd ~/../yy-volume/x2ct/imbalanced-dataset-sampler
# python setup.py install
 
# cd ~/../yy-volume/codes
# CUDA_VISIBLE_DEVICES=0,1 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-8 -num 12
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-5 -num 13 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-2 -num 14 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-3 -num 15 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-5 -wd 1e-8 -num 8
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-5 -wd 1e-5 -num 9
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-5 -wd 1e-2 -num 10
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-5 -wd 1e-1 -num 11
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-4 -wd 1e-4 -num 0 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-4 -wd 1e-3 -num 1 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-4 -wd 1e-2 -num 2 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 5e-4 -wd 1e-1 -num 3 
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-5 -wd 1e-4 -num 4
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-5 -wd 1e-3 -num 5
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-5 -wd 1e-2 -num 6
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-5 -wd 1e-1 -num 7


# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-1 -num 0
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 5e-1 -num 1
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1 -num 2
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-4 -num 3
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-7 -wd 1e-1 -num 4
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-7 -wd 1e-2 -num 5
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-7 -wd 1e-4 -num 6
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-7 -wd 1e-3 -num 7
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-5 -wd 1e-4 -num 8
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-5 -wd 1e-3 -num 9
# cd ~/../yy-volume/x2ct/DANN_py3_v2/logs/
# rm -r *
# cd ~/../yy-volume/codes
# python ~/../yy-volume/x2ct/DANN_py3_v2/main.py  -lr 1e-6 -wd 1e-1 -num 12
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-1 -num 10
# python ~/../yy-volume/x2ct/DANN_py3/main.py  -lr 1e-6 -wd 1e-2 -num 11
