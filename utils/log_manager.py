import argparse
parser = argparse.ArgumentParser(description='adsl log process programme')
parser.add_argument('--rootDir', required=True, default=None, help='root directory location where the logs stored in')
args_opt, unknown = parser.parse_known_args()



import os
filepaths = []
for _,_,files in os.walk(args_opt.rootDir):
    for filename in files:
        filepaths.append(os.path.join(args_opt.rootDir,filename))
    break


import re
import json
from CSV import CSV
perfdict = dict()
for filename in filepaths:
    with open(filename,'r') as fp:
        contents = fp.readlines()
        for line in contents:
            if '[BEGINE]' in line:
                begin1=True
                begin2=True
                
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
                perfdict[key]['performance']['speed']=[]
                perfdict[key]['performance']['throughput']=[]
                continue
            if '[auto mtf search]' in line:
                pattern = r'\([^:()]*\)'
                perfdict[key]['strategy'] = re.findall(pattern,line)
                continue
            if 'INFO:tensorflow:acc = ' in line:
                pattern = r'[0-9\.]{1,}'
                if begin1:
                    perfdict[key]['performance']['accuracy']=[]
                    perfdict[key]['performance']['loss']=[]
                    begin1=False
                else:
                    accuracy,loss,_ = re.findall(pattern,line)
                    accuracy=float(accuracy)
                    loss=float(loss)
                    perfdict[key]['performance']['accuracy'].append(accuracy)
                    perfdict[key]['performance']['loss'].append(loss)
                continue
            if 'INFO:tensorflow:loss =' in line:
                pattern = r'[0-9\.]{1,}'
                if begin2:
                    perfdict[key]['performance']['step']=[]
                    begin2=False
                else:
                    _,step,_ = re.findall(pattern,line)
                    step=int(step)
                    perfdict[key]['performance']['step'].append(step)
                continue
            if 'INFO:tensorflow:global_step/sec:' in line:
                pattern = r'[0-9\.]{1,}'
                speed = re.findall(pattern,line)[0]
                speed = float(speed)
                throughput = speed*int(config['batch_size'])
                speed = 1000/speed
                perfdict[key]['performance']['speed'].append(speed)
                perfdict[key]['performance']['throughput'].append(throughput)
                continue
            

for key, value in perfdict.items():
    subpath = os.path.join(args_opt.rootDir,'output/'+key)
    if os.path.exists(subpath):
        import shutil
        # if exited, remove it!
        shutil.rmtree(subpath)
    os.makedirs(subpath,exist_ok=True)
    strategy_path = os.path.join(subpath,'strategy')

    csv_abspath = re.sub(r'_device_num_\d', '-log.csv', key)
    csv_path = os.path.join(subpath,csv_abspath)
    with open(strategy_path,'w') as fp:
        for item in value['strategy']:
            fp.write(item+'\n')
    if 'step' in value['performance'].keys() and\
        'speed' in value['performance'].keys() and\
        'throughput' in value['performance'].keys() and\
        'loss' in value['performance'].keys():
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

