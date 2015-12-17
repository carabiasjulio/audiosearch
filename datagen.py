from matplotlib import pyplot as pl
from scipy import signal 
import scikits.audiolab as audiolab
import matplotlib.pyplot as pl
import os
import numpy as np
import csv
import VectorQuantizer as vq
import HartiganOnline as hartigan
import spectrogram



data_dir = '/home/mzhan/audiosearch/data/UrbanSound8K/audio' 
metadata_path = '/home/mzhan/audiosearch/data/UrbanSound8K/metadata/UrbanSound8K_by_fold.csv'

errorfiles = []
def data_gen(verbose=False):
    """ 
    Generate spectrogram for each audio file (in log amplitude). 
    Yields numpy array of shape (n,d) where n is number of windows (number of frames in file) and 
    d is STFT resolution (frame dimension).
    """
    SR = 44100   # standard sampling rate
    with open(metadata_path, 'rb') as fmeta:
        meta = csv.reader(fmeta, delimiter=',')
        meta.next()   # skip header
        for line in meta:
            fname, fsID, start, end, salience, fold, classID, className = line
            if verbose:
                print fname, fold
            try: 
                s = audiolab.Sndfile(os.path.join(data_dir, 'fold'+str(fold), fname))
                w = s.read_frames(s.nframes)    # bitdepth: float in range [-1,1]
                if len(w.shape)>1:
                    w = w[:,0]    # take the first channel when multiple are present  
                if s.samplerate==SR:
                    resampled = w
                else:
                    resampled = signal.resample(w, 1.*s.nframes/s.samplerate*SR)   # unify sample rates
                frameSize = SR*0.02   # frame duration = 20ms
                S = spectrogram.stft(resampled, frameSize)
                yield np.log(np.abs(S))
            except:
                errorfiles.append((fold, fname))   # ignore files that can't be opened by wave package

def snd_gen(verbose=False):
    """ 
    Generate audio files as audiolab.Sndfile classes
    """
    with open(metadata_path, 'rb') as fmeta:
        meta = csv.reader(fmeta, delimiter=',')
        meta.next()   # skip header
        for line in meta:
            fname, fsID, start, end, salience, fold, classID, className = line
            if verbose:
                print fname, fold
            try: 
                s = audiolab.Sndfile(os.path.join(data_dir, 'fold'+str(fold), fname))
                yield s
            except:
                errorfiles.append((fold, fname))   # ignore files that can't be opened by wave package

                
