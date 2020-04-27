python main_covid.py --epochs 150 -j 8 --logpath logs/log_covid_1/ --resumedir checkpoints/checkpoints_covid_1/ --wd 3e-3  --lr 0.0001  -a resnet101  ../datasets/covid --gpu 1 --shellnum 0  &&  
python main_covid.py --epochs 150 -j 8 --logpath logs/log_covid_1/ --resumedir checkpoints/checkpoints_covid_1/ --wd 3e-3  --lr 0.001  -a resnet50  ../datasets/covid --gpu 1 --shellnum 1     








