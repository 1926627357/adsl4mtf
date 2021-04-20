rm -rf output/
python run_pretraining.py --mode=data_parallel --bert_config_file=./bert_config.json --input_train_files=/data/DNN_Dataset/dataset/zhwiki/tfdir/ --output_dir=./output --gpu_num=2 &> 2.log
rm -rf output/
python run_pretraining.py --mode=data_parallel --bert_config_file=./bert_config.json --input_train_files=/data/DNN_Dataset/dataset/zhwiki/tfdir/ --output_dir=./output --gpu_num=1 &> 1.log
rm -rf output/
python run_pretraining.py --mode=data_parallel --bert_config_file=./bert_config.json --input_train_files=/data/DNN_Dataset/dataset/zhwiki/tfdir/ --output_dir=./output --gpu_num=4 &> 4.log
rm -rf output/