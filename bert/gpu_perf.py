


import subprocess
import time
import signal
import pandas as pd

def my_handler(signum, frame):
    global stop
    stop = True
    print("Stop process ...")
 
 
# 设置相应信号处理的handler
signal.signal(signal.SIGINT, my_handler)
signal.signal(signal.SIGHUP, my_handler)
signal.signal(signal.SIGTERM, my_handler)


stop = False
result=[]
interval = 5
gpu_num = 4
while True:
    try:
        if stop:
            # 中断时需要处理的代码
            for i in range(gpu_num):
                start = 0    
                pd.DataFrame(columns=['gpu usage(%)','mem usage(%)','mem usage(MB)'],data=result[i:len(result):gpu_num])\
                    .to_csv('./gpu{}_{}.csv'.format(i,interval))
                

                
            break
        else:
            cmd = f'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv | tail -n {gpu_num}'
            output = subprocess.run(cmd,stdout=subprocess.PIPE,shell=True)
            result += [each.split(',') for each in output.stdout.decode("utf-8").split('\n')[:-1]]
    except Exception as e:
        print(str(e))
        break
    time.sleep(interval)
        

