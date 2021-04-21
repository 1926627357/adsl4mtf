rm -rf output/
python run_pretraining.py --mode=data_parallel --bert_config_file=./bert_config.json --input_train_files=/data/DNN_Dataset/zhwiki/tfdir/ --output_dir=./output --gpu_num=2 &> 2.log
echo "gpu-2 done"
rm -rf output/
python run_pretraining.py --mode=data_parallel --bert_config_file=./bert_config.json --input_train_files=/data/DNN_Dataset/zhwiki/tfdir/ --output_dir=./output --gpu_num=1 &> 1.log
echo "gpu-1 done"
rm -rf output/
python run_pretraining.py --mode=data_parallel --bert_config_file=./bert_config.json --input_train_files=/data/DNN_Dataset/zhwiki/tfdir/ --output_dir=./output --gpu_num=4 &> 4.log
echo "gpu-4 done"
rm -rf output/