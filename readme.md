- Have finished the main.py(can use command params like following):

train smallNet command 
pure mri:
python main.py --usesmall 1 --logpath logs/log_smallNet_puremri/ --traindatapath ../datasets/mr_ct_raw_dataset/trainA/ --testdatapath ../datasets/mr_ct_raw_dataset/testA/ --augement expe --epochs 1000 --lr 5e-6 -b 128 datasets

fake ct:
python main.py --usesmall 1 --logpath logs/log_smallNet_fakect/ --traindatapath ../datasets/mr_ct_trans_dataset/fake_ct_train/ --testdatapath ../datasets/mr_ct_trans_dataset/fake_ct_test/ --augement expe --epochs 1000 --lr 5e-6 -b 128 datasets

