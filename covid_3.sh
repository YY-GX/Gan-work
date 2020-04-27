python main_covid.py --epochs 150 -j 8 --logpath logs/log_covid_3/ --resumedir checkpoints/checkpoints_covid_3/ --wd 1e-2  --lr 0.0001  -a resnet101  ../datasets/covid --gpu 3 --shellnum 0  &&  
python main_covid.py --epochs 150 -j 8 --logpath logs/log_covid_3/ --resumedir checkpoints/checkpoints_covid_3/ --wd 1e-2  --lr 0.001  -a resnet50  ../datasets/covid --gpu 3 --shellnum 1    















