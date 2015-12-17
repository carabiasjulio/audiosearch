from matplotlib import pyplot as pl
from scipy import signal 
import wave

data_dir = '/home/mzhan/audiosearch/data/UrbanSound8K/audio' 
metadata = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'

### 1. Preprocessing

fold = 1
fname = '/'.join([data_dir, 'fold'+str(fold), '7061-6-0-0.wav'])
f = wave.open(fname)
sf = f.getframerate()   # sampling frequency
print sf
n = f.getnframes()
wav = f.readframes(n)
print wav
pl.plot(wav)
pl.show()
spec = signal.spectrogram(wav, fs=sf, window=('hanning'), nperseg=256, noverlap=128)




### convert audio data from waveform to spectrogram representation


### Vector Quantization on audio sample spectrograms using kmeans

# find cluster centers
#
# fit 

