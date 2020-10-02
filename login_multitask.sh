#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 
cd ~/../yy-volume/codes/

# # Param tune~

# # Test multi -> -5
# # check whether the same as PURE
# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o1_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o1_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0 --lambda2 1   datasets 

# # Test multi1 -> -4

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o2_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o2_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 1e-1 --lambda2 1 --mymodel 1  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o3_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o3_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9  --mymodel 1  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o4_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o4_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 1  datasets 

# # Test multi2 -> -3
# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o5_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o5_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95  --mymodel 2  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o6_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o6_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99  --mymodel 2  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o7_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o7_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.001 --lambda2 0.999  --mymodel 2  datasets 


# Test multi -> -4
# pure作为pretrain的-4：跑的还行的加上pure的pretrian
# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o1_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o1_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 1   --mymodel 3  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o2_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o2_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 3  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o3_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o3_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 3  datasets 
# # Test multi1 -> -3：-3 01 0.9之前这个跑的效果还行，加pretrain看看

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o4_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o4_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o5_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o5_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets # BEST

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o6_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o6_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 64 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 1  datasets 

# # Test multi2 -> -5: -5层，之前没跑
# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o7_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o7_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o8_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o8_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o9_v2/ --resumedir checkpoints/checkpoints_baseline_multitask/o9_v2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.001 --lambda2 0.999  datasets 











# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v1 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v1 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.5 --lambda2 0.5 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v2 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v2 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.4 --lambda2 0.6   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v3 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v3 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.3 --lambda2 0.7   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v4 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v4 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.2 --lambda2 0.8   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v5 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v5 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v6 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v6 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.7 --lambda2 0.3   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v7 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v7 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --wd 1e-6 --lambda1 0.5 --lambda2 0.5   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v8 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v8 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --wd 1e-5 --lambda1 0.5 --lambda2 0.5   datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/v9 --resumedir checkpoints/checkpoints_baseline_multitask/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/v9 --testdatapath ../datasets/final_mr_ct/testB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --wd 1e-4 --lambda1 0.5 --lambda2 0.5   datasets 


# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o1/ --resumedir checkpoints/checkpoints_baseline_multitask/o1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o2/ --resumedir checkpoints/checkpoints_baseline_multitask/o2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99    --mymodel 4 --wd 1e-6 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o3/ --resumedir checkpoints/checkpoints_baseline_multitask/o3/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  --wd 1e-7 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o4/ --resumedir checkpoints/checkpoints_baseline_multitask/o4/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95    --mymodel 4 datasets 

# # Run results

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o1/ --resumedir checkpoints/checkpoints_baseline_multitask/o1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o2/ --resumedir checkpoints/checkpoints_baseline_multitask/o2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9    --mymodel 4 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o3/ --resumedir checkpoints/checkpoints_baseline_multitask/o3/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o4/ --resumedir checkpoints/checkpoints_baseline_multitask/o4/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9    --mymodel 4 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o5/ --resumedir checkpoints/checkpoints_baseline_multitask/o5/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9    --mymodel 4 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o6/ --resumedir checkpoints/checkpoints_baseline_multitask/o6/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o7/ --resumedir checkpoints/checkpoints_baseline_multitask/o7/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o8/ --resumedir checkpoints/checkpoints_baseline_multitask/o8/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9    --mymodel 4 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o9/ --resumedir checkpoints/checkpoints_baseline_multitask/o9/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o10/ --resumedir checkpoints/checkpoints_baseline_multitask/o10/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o11/ --resumedir checkpoints/checkpoints_baseline_multitask/o11/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9    --mymodel 4 datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o12/ --resumedir checkpoints/checkpoints_baseline_multitask/o12/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9    --mymodel 4 datasets 


# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o1_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o1_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o2_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o2_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o3_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o3_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o4_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o4_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o5_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o5_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o6_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o6_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o7_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o7_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o8_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o8_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o9_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o9_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o10_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o10_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o11_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o11_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_baseline_multitask/o12_final/ --resumedir checkpoints/checkpoints_baseline_multitask/o12_final/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 50 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 











# PET RUN 0.1 0.9
# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o1_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o1_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o2_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o2_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o3_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o3_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o4_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o4_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o5_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o5_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o6_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o6_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o7_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o7_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o8_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o8_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o9_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o9_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o10_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o10_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o11_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o11_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o12_pet_09/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o12_pet_09/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 


# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o1/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o1/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.01 --lambda2 0.99   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o2/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o2/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.1 --lambda2 0.9   --mymodel 4  datasets 

# python main_times_multitask.py -a resnet50 --logpath logs/log_pet_baseline_multitask/o3/ --resumedir checkpoints/checkpoints_pet_baseline_multitask/o3/ --filename pure.csv --traindatapath ../datasets/final_mr_ct/trainA/ --testdatapath ../datasets/final_mr_ct/valB/ --augement pure  --epochs 60 --times 1  -b 32 --dropout 0.5 --lr 1e-4 --lambda1 0.05 --lambda2 0.95   --mymodel 4  datasets 