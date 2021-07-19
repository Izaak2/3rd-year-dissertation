"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Racording a commands dataset.
Description:
"""

import sounddevice as sd
import soundfile as sf
import librosa
import time
from random import randint


def recordCommand(filename):
    samplerate = 8000
    duration = 1 # seconds
    filename = filename
    print('\a')
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    print('\a')
    sd.wait()
    sf.write(filename, mydata, samplerate)

def getCommand(filename, sample_rate=8000):
    filename = filename
    samples, sample_rate = librosa.load(filename, sr = sample_rate)
    samples = librosa.resample(samples, sample_rate, 8000)
    mfcc = librosa.feature.mfcc(y=samples, sr=8000, n_mfcc=13, dct_type=2, norm='ortho', lifter=26)
    mfcc = mfcc.flatten()
    mfcc = mfcc.reshape(1, 13*16)
    return mfcc
