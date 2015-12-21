### Feature extraction on existing sound datasets

import os, fnmatch
import numpy as np
import random

# gather all files
random.seed(1228)
data_root='data'
dataset_names = os.listdir(data_root)
def scan_datasets(shuffle=True):
    dataset = []
    for dataset_name in dataset_names:
        samples = []
        for root, dirs, fs in os.walk(os.path.join('data', dataset_name)):
            for f in fnmatch.filter(fs, '*.wav'):
                samples.append(os.path.join(root, f))
        if shuffle:
            random.shuffle(samples)
        with open(dataset_name+'.txt','wb') as f:
            f.write('\n'.join(samples))
        dataset.append(samples)
    return dataset  

#datasets = scan_datasets()


from utils import *
def extract_dataset(dataset_name):
    with open(dataset_name+'.txt') as f:
        samples = f.readlines()
    X = np.zeros((len(samples),D))
    for i in range(len(samples)):
        s = samples[i]
        X[i] = get_features(s.strip('\n')).reshape((1,D))
        if not i%10:
            print "%d files processed" % i
    return X

