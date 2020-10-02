#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 
cd ~/../yy-volume/codes/

# # Param tune~


# PET
# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o1/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o2/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o3/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o3/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 


# 正式跑的
# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o1_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o1_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o2_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o2_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o3_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o3_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o4_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o4_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o5_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o5_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o6_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o6_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o7_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o7_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o8_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o8_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o9_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o9_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o10_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o10_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o11_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o11_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o12_pet/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o12_pet/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 
