python main_covid.py --epochs 150 -j 8 --logpath logs/log_covid/ --resumedir checkpoints/checkpoints_covid/ --wd 1e-3  --lr 0.0001  -a resnet101  ../datasets/covid --gpu 0 --shellnum 0 &&       
python main_covid.py --epochs 300 -j 8 --logpath logs/log_covid/ --resumedir checkpoints/checkpoints_covid/ --wd 1e-3  --lr 0.001  -a resnet50  ../datasets/covid --gpu 0 --shellnum 1         








