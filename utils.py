import random
import wave
import os
from matplotlib import pyplot as pl
from scipy import signal 
import soundfile as sf
import sounddevice2 as sd
import numpy as np
import csv
import VectorQuantizer as vq
import HartiganOnline as hartigan
import spectrogram

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

# Load library
datasets = os.listdir('data')
dataset_choice = 3 
dataset = datasets[dataset_choice]
print 'Dataset:', dataset

X = np.load(dataset+'.npy')
N,D = X.shape
with open(dataset+'.txt') as f:
    LIBSAMPLE_PATHS = np.array( ''.join(f.readlines()).split('\n'))
Y = np.load(dataset+'_Y.npy')
CLASS_NAMES = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine idling", "gun shot", "jackhammer", "siren", "street music"]

def get_random_libsample():
    ''' Get a random library sample, as file name string '''
    n = random.randint(0,N)
    return LIBSAMPLE_PATHS[n]

def show_wav(w, sr=44100):
    print len(w), sr
    if len(w.shape)>1:
        w = w[:,0]    # take only 1 channel if stereo
    pl.figure()
    pl.plot(w)
    pl.ylim([-1,1])
    frameSize = sr*0.02
    S = spectrogram.stft(w, frameSize)
    print S.shape
    pl.figure()
    pl.imshow(np.log(abs(S).T), origin="lower", aspect="auto", interpolation="none")

def play_file(f):
    w,fs = sf.read(f)
    sd.play(w,fs)

def play_sample(i):
    play_file(LIBSAMPLE_PATHS[i])

# vector quantizer for feature extraction
NC = 512	# number of features (centroids)
quantizer = vq.VectorQuantizer(clusterer=hartigan.HartiganOnline(n_clusters=NC))
# load trained quantizer
centroids = np.load('/home/mzhan/audiosearch/centroids151209.npy')
quantizer.load_centroids(centroids[0,7])
# use only a subset of features
feature_inds = np.load('feature_inds.npy')


def show_centroids(centroids):
    ''' Display quantizer centroids (components) of shape (n_centroids, n_data_dimensions)'''
    pl.figure(figsize=(22,6))
    pl.imshow(centroids.T, aspect="auto", interpolation="none")
    pl.colorbar()


def get_features(f, reduced=True):
    """ Feature extraction
    f: string, audio sample file name
    """
    w, fs = sf.read(f)
    # convert to log spectrogram
    logS = waveToLogSpec(w,fs)
    # quantize
    quantized = quantizer.transform(logS)
    bins = quantized.A.sum(0)    
    # normalize feature counts to distribution
    normalized = bins/bins.sum()
    # dimension reduction
    if reduce:
        x = normalized[feature_inds]
    return x

def waveToLogSpec(w, fs):
    SR = 44100
    if fs==SR:
        resampled = w
    else:
        resampled = signal.resample(w, 1.*len(w)/fs*SR)
    frameSize = SR*0.02   # frame duration = 20ms
    if len(w.shape)>1:
        w = w[:,0]    # take the first channel when multiple are present   
    S = spectrogram.stft(resampled, frameSize)
    logS = np.log(np.abs(S))
    return logS



