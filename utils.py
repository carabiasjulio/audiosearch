import random
import pyaudio  
import wave
import os
  

data_dir = '/home/mzhan/audiosearch/data/UrbanSound8K/audio' 
metadata_path = '/home/mzhan/audiosearch/data/UrbanSound8K/metadata/UrbanSound8K_by_fold.csv'
feature_path = '/home/mzhan/audiosearch/data/UrbanSound8K/features2.npy'


def random_audio_file():
    fold = random.choice(os.listdir(data_dir))
    if fold[:4]!='fold':
        return random_audio_file()
    fname = random.choice(os.listdir(os.path.join(data_dir, fold)))
    return os.path.join(data_dir, fold, fname)

def play_wave(fname):
    try:  # Use the wave package to read
        #define stream chunk   
        chunk = 1024  

        #open a wav format music  
        f = wave.open(fname,"rb")  
        #instantiate PyAudio  
        p = pyaudio.PyAudio()  
        #open stream  
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                        rate = f.getframerate(),  
                        output = True)  
        #read data  
        data = f.readframes(chunk)  

        #paly stream  
        while data != '':  
            stream.write(data)  
            data = f.readframes(chunk)  

        #stop stream  
        stream.stop_stream()  
        stream.close()  

        #close PyAudio  
        p.terminate()  
    except Exception as e:
        print 'Cannot open', fname
    
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
