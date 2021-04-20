import os
import subprocess

def excuteCommand(com):
    ex = subprocess.Popen(com, stdout=subprocess.PIPE, shell=True)
    out, err  = ex.communicate()
    status = ex.wait()
    print("cmd in:", com)
    print("cmd out: ", out.decode())
    return out.decode()
count=0
for root,_,files in os.walk('/data/haiqwa-dataset/dataset/zhwiki/text'):
    for each_file in files:
        count+=1
        
        excuteCommand(f'/home/haiqwa/anaconda3/envs/mtf/bin/python /data/haiqwa-dataset/dataset/zhwiki/bert/create_pretraining_data.py \
                    --input_file={os.path.join(root,each_file)} \
                    --output_file=/data/haiqwa-dataset/dataset/zhwiki/tfdir/{count}.tfrecord \
                    --vocab_file=/data/haiqwa-dataset/dataset/zhwiki/chinese_L-12_H-768_A-12/vocab.txt\
                    --do_lower_case=True \
                    --max_seq_length=128 \
                    --max_predictions_per_seq=20 \
                    --masked_lm_prob=0.15 \
                    --random_seed=12345 \
                    --dupe_factor=5') 
        print(os.path.join(root,each_file)," complete")
        
    


