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

# Load library
datasets = os.listdir('data')
dataset_choice = 1
dataset = datasets[dataset_choice]
print 'Dataset:', dataset

X = np.load(dataset+'.npy')
N,D = X.shape
with open(dataset+'.txt') as f:
    LIBSAMPLE_PATHS = np.array( ''.join(f.readlines()).split('\n'))

# number of library samples (=8732)

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


def get_features(f):
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



