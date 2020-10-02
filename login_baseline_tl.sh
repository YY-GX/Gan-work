#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 
cd ~/../yy-volume/codes/ 
# python main_times.py -a resnet18 --logpath logs/log_baseline_tl/v1/ --resumedir checkpoints/checkpoints_baseline_tl/v1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0.95 --lr 1e-4  datasets --wd 0
# python main_times.py -a resnet18 --logpath logs/log_baseline_tl/v2/ --resumedir checkpoints/checkpoints_baseline_tl/v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0.9 --lr 1e-4  datasets --wd 0
# python main_times.py -a resnet18 --logpath logs/log_baseline_tl/v3/ --resumedir checkpoints/checkpoints_baseline_tl/v3/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0.85 --lr 1e-4  datasets --wd 0
# python main_times.py -a resnet18 --logpath logs/log_baseline_tl/v4/ --resumedir checkpoints/checkpoints_baseline_tl/v4/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0.8 --lr 1e-4  datasets --wd 0
# python main_times.py -a resnet18 --logpath logs/log_baseline_tl/v5/ --resumedir checkpoints/checkpoints_baseline_tl/v5/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0.75 --lr 1e-4  datasets --wd 0
# python main_times.py -a resnet18 --logpath logs/log_baseline_tl/v6/ --resumedir checkpoints/checkpoints_baseline_tl/v6/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0.7 --lr 1e-4  datasets --wd 0

# python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v1/ --resumedir checkpoints/checkpoints_baseline_tl/v1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0 --lr 1e-4  datasets --wd 1e-8

# python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v2/ --resumedir checkpoints/checkpoints_baseline_tl/v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valA/ --augement mr_pure  --epochs 40 --times 1  -b 64 --dropout 0 --lr 1e-4  datasets --wd 0

python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v1/ --resumedir checkpoints/checkpoints_baseline_tl/v1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/pet_train/ --testdatapath ../datasets/final_mr_ct/pet_val/ --augement pet_pure  --epochs 50 --times 1  -b 64 --dropout 0 --lr 1e-4  datasets --wd 0

python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v2/ --resumedir checkpoints/checkpoints_baseline_tl/v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/pet_train/ --testdatapath ../datasets/final_mr_ct/pet_val/ --augement pet_pure  --epochs 50 --times 1  -b 64 --dropout 0 --lr 1e-3  datasets --wd 0

python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v3/ --resumedir checkpoints/checkpoints_baseline_tl/v3/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/pet_train/ --testdatapath ../datasets/final_mr_ct/pet_val/ --augement pet_pure  --epochs 50 --times 1  -b 64 --dropout 0 --lr 1e-5  datasets --wd 0

python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v4/ --resumedir checkpoints/checkpoints_baseline_tl/v4/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/pet_train/ --testdatapath ../datasets/final_mr_ct/pet_val/ --augement pet_pure  --epochs 50 --times 1  -b 32 --dropout 0 --lr 1e-4  datasets --wd 0

python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v5/ --resumedir checkpoints/checkpoints_baseline_tl/v5/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/pet_train/ --testdatapath ../datasets/final_mr_ct/pet_val/ --augement pet_pure  --epochs 50 --times 1  -b 32 --dropout 0 --lr 1e-3  datasets --wd 0

python main_times.py -a resnet50 --logpath logs/log_baseline_tl/v6/ --resumedir checkpoints/checkpoints_baseline_tl/v6/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/pet_train/ --testdatapath ../datasets/final_mr_ct/pet_val/ --augement pet_pure  --epochs 50 --times 1  -b 32 --dropout 0 --lr 1e-5  datasets --wd 0