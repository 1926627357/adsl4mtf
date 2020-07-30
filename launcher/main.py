import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
'''
ERROR: 3
WARNING: 2
INFO: 1
DEBUG: 0
'''
os.environ['GLOG_v'] = '0'
sys.path.append(os.path.abspath('..'))


import mesh_tensorflow as mtf
from adsl4mtf.dataset import load_dataset  # local file import
import tensorflow.compat.v1 as tf
from adsl4mtf.model_zoo import network


import argparse

parser = argparse.ArgumentParser(description='ResNet50 train.')

