import argparse
parser = argparse.ArgumentParser(description='adsl log process programme')
parser.add_argument('--rootDir', required=True, default=None, help='root directory location where the logs stored in')
args_opt, unknown = parser.parse_known_args()



import os
filepaths = []
for _,_,files in os.walk(args_opt.rootDir):
    for filename in files:
        filepaths.append(os.path.join(args_opt.rootDir,filename))


import re
import json
from CSV import CSV
perfdict = dict()
for filename in filepaths:
    with open(filename,'r') as fp:
        contents = fp.readlines()
        for line in contents:
            if '[BEGINE]' in line:
                step=0
                continue
            if '[Input args]' in line:
                pattern = r'\{.*\}'
                line = re.findall(pattern,line)[0]
                line = line.replace("\'","\"")
                config = json.loads(line)
                continue
            if '[configuration]' in line:
                pattern = r'model_.*'
                key = re.findall(pattern,line)[0]
                perfdict[key] = dict()
                perfdict[key]['performance'] = dict()
                continue
            if '[auto mtf search]' in line:
                pattern = r'\([^:()]*\)'
                perfdict[key]['strategy'] = re.findall(pattern,line)[0]
                continue
            if 'INFO:tensorflow:acc = ' in line:
                pattern = r'[0-9\.]{1,}'
                accuracy,loss,_ = re.findall(pattern,line)
                accuracy=float(accuracy)
                loss=float(loss)
                if step==0:
                    perfdict[key]['performance']['accuracy']=[]
                    perfdict[key]['performance']['loss']=[]
                else:
                    perfdict[key]['performance']['accuracy'].append(accuracy)
                    perfdict[key]['performance']['loss'].append(loss)
                continue
            if 'INFO:tensorflow:loss =' in line:
                pattern = r'[0-9\.]{1,}'
                _,step,_ = re.findall(pattern,line)
                step=int(step)
                if step==0:
                    perfdict[key]['performance']['step']=[]
                else:
                    perfdict[key]['performance']['step'].append(step)
                continue
            if 'INFO:tensorflow:global_step/sec:' in line:
                pattern = r'[0-9\.]{1,}'
                speed = re.findall(pattern,line)[0]
                speed = float(speed)
                throughput = speed*int(config['num_gpus'])*int(config['batch_size'])
                if step==0:
                    perfdict[key]['performance']['speed']=[]
                    perfdict[key]['performance']['throughput']=[]
                else:
                    perfdict[key]['performance']['speed'].append(speed)
                    perfdict[key]['performance']['throughput'].append(throughput)
                continue

for key, value in perfdict.items():
    subpath = os.path.join(args_opt.rootDir,key)
    strategy_path = os.path.join(subpath,'strategy')
    csv_path = os.path.join(subpath,key+'-log.csv')
    with open(strategy_path,'w') as fp:
        for item in value['strategy']
            fp.write(item+'\n')
    csvfile = CSV(
                    path=csv_path,
                    columns=['step','step_cost_time(ms)','samples/second','loss'],
                    values=[
                        value['performance']['step'],
                        value['performance']['speed'],
                        value['performance']['throughput'],
                        value['performance']['loss'],
                    ])
    csvfile.dump()

# print(blocks[1])

