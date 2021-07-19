
# Author: Kacper Wojtasik 2020

import numpy as np
from scipy.fftpack import dct
import signal_processing as sp
import numpy

def mfcc_extraction(signal, sample_rate=16000, window_length=0.025, window_step=0.01,
                    mfcc_number=13, filter_number=26, fft_size=None, low_freq=0,
                    high_freq=None, preemph=0.97, ceplifter=22, append_energy=True,
                    mean_normalization=False, normalize_amplitude=False ,window_function=lambda x:numpy.ones((x,))):

    """
         Extract MFFC features from given a signal.

         Paramteres
         signal: audio signal from which MFCC features will be extracted
         sample_rate: sample rate of the signal
         window_length: the length in seconds of the analysis window
         window_step: step between windows in seconds
         mfcc_number: the number of extracted features
         filter_number: number of filters in filter bank
         fft_size: FFT size
         low_freq: lower cut frequency for mel filters in Hz
         high_freq: high cut frequency for mel filters in Hz
         preemph: coeeficient of preemphasizing process
         ceplifter: number of order of liftering filter. 0 no filter
         append_energy: if true, 0th cepstral coefficient is swaped with the log of the total frame energy
         window_function: e.g. winfunc=numpy.hamming
         mean_normalization: if true power spectrum of frames will be mean normalized
         normalize_amplitude: if true the signal amplitude is normalize from -1 to 1
         returns: a numpy array with shape [number of frames by number of mfcc coefficeints]. Every row holds a feature vetor
     """
    if normalize_amplitude:
        signal = signal / np.max(np.abs(signal))
        # Computing fft_size
    fft_size = fft_size or sp.calculate_fft_size(sample_rate, window_length)
        # Getting filtered framed signal
    features, energy = filter_bank(signal, sample_rate, window_length, window_step, filter_number, fft_size, low_freq, high_freq, preemph, mean_normalization, window_function)
        # Changing values into dBs
    features = np.log(features)
        # Extraction of MFCC vectors 
    mfcc_features = dct(features, type=2, axis=1, norm='ortho')[:,:mfcc_number]
        # Liftering of mfcc to ncrease SNR
    mfcc_features = sp.cepstral_liftering(mfcc_features, ceplifter)
        # first cepstral is swaped with log of frame energy
    if append_energy:
        mfcc_features[:,0] = np.log(energy)
    return mfcc_features

def filter_bank(signal, sample_rate=16000, window_length=0.02, window_step=0.01,
                filter_number=26, fft_size=512, low_freq=0, high_freq=None,
                preemph=0.97, mean_normalization=False, window_function=lambda x:np.ones((x,))):
    """
    Calculate Mel filterbank energy features.

    Paramteres
    signal: audio signal from which MFCC features will be extracted
    sample_rate: sample rate of the signal
    window_length: the length in seconds of the analysis window
    window_step: step between windows in seconds
    mfcc_number: the number of extracted features
    filter_number: number of filters in filter bank
    fft_size: FFT size
    low_freq: lower cut frequency for mel filters in Hz
    high_freq: high cut frequency for mel filters in Hz
    preemph: coeeficient of preemphasizing process
    window_function: e.g. winfunc=numpy.hamming
    mean_normalization: if true power spectrum of frames will be mean normalized
    return:
    -   nparray of size (number frames by filter number) containging features.
        In every row there is 1 feature vector.
    -   energy in each frame
    """
    high_freq = high_freq or sample_rate / 2
    signal = sp.preemhasis(signal, preemph)
    frames = sp.frame_signal(signal, window_length * sample_rate, window_step * sample_rate, window_function)
    power_spectrum = sp.power_spectrum(frames, fft_size)
    """
    if mean_normalization:
        power_spectrum -= (np.mean(power_spectrum, axis=0) + 1e-8)
    """
    energy = np.sum(power_spectrum, 1) # stores total energy in each frame
    energy = np.where(energy == 0, np.finfo(float).eps, energy) # energy equals to 0 causes probelems with log

    filter_bank = sp.get_filterbanks(filter_number, fft_size, sample_rate, low_freq, high_freq)
    features = np.dot(power_spectrum, filter_bank.T) # calculating filter bank energies
    features = np.where(features == 0, np.finfo(float).eps, features) # 0 causes problems with log

    return features, energy
