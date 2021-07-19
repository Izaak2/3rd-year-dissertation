
# Author: Kacper Wojtasik 2020

import decimal
import numpy as np
import scipy.fftpack as dct
import math
import logging
import numpy

#   converts frequency in Hz into mels
def freq_to_mel(freq):
    """
    converts frequency in Hz into mels

    Paramteres:
    freq: a value of frequency to be converted
    returns: converted frequency
    """
    return 2595.0 * np.log10(1.0 + freq / 700.0)

#   converts mels to frequency in Hz
def mel_to_freq(mel):
    """
    converts mels to frequency

    Parameters:
    mel: a value of frequency to be converted
    returns: converted mels
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def rolling_window(signal, window, step=1):
    shape = signal.shape[:-1] + (signal.shape[-1] - window + 1, window)
    strides = signal.strides + (signal.strides[-1],)
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)[::step]

def round_half_up(number):
    """
    Rounding a number with mathematical rules

    Parameters:
    number: number to be rounded
    return: rounded integer
    """
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def calculate_fft_size(sample_rate=16000, window_length=0.1):
    """
    Return FFT size.

    Paramters:
    sample_rate: sample rate of a signat that fft will be applied
    window_length: duration of a window in seconds
    """
    # converting window length to samples
    window_length *= sample_rate

    # setting fft_size to 1
    fft_size = 1

    # multiplying by 2 fft_size as long as it will be longer or equal than window_length
    while fft_size < window_length:
        fft_size *= 2

    return fft_size

def preemhasis(signal, coefficient=0.97):
    """
    Preemhasising the input signal.

    Parameters:
    singal: signal to be preemphasized
    coefficient: coefficient for the process.
    returns an preemphisized signal.
    """
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

def frame_signal(signal, frame_length, frame_step, window_function=lambda x:np.ones((x,)), stride_trick=True):
    """
    Framing signal into ovelapping frames.

    Paramters:
    signal: signal to be framed
    frame_length: frame length in samples
    frame_step: number of samles after the start of previous frame that the new frame should start
    window_function: a window to applied for window process
    stride_trick: speeding up computing time
    returns: numpy array number frames by frame length
    """
    # Converting the parameters into samples
    signal_length = len(signal)
    frame_length = int(round_half_up(frame_length))
    frame_step = int(round_half_up(frame_step))
    overlap_length = int(round_half_up(frame_length - frame_step))

    # Computing the total number of frames
    if signal_length <= frame_length:
        frame_number = 1
    else:
        frame_number = np.abs((signal_length - overlap_length)) // abs((frame_length - overlap_length))

    # Checking if there are samples not inclueded into frames. 
    rest_samples = np.abs(signal_length - overlap_length) % np.abs(frame_length - overlap_length)

    if rest_samples != 0:   # padding the signal with zeros
        pad_signal_length = int(frame_step - rest_samples)
        lacking_samples = np.zeros(pad_signal_length)
        pad_signal = np.append(signal,lacking_samples)
        frame_number += 1
    else:
        pad_signal = signal

    if stride_trick: # speeding up calculations
        window = window_function(frame_length)
        frames = rolling_window(pad_signal, window=frame_length, step=frame_step)
    else:   # compute frames with the maths equation
        ind1 =  np.tile(np.arange(0, frame_length), (frame_number, 1))
        ind2 = np.tile(np.arange(0,frame_number * frame_step, frame_step), (frame_length, 1)).T
        indicies = ind1 + ind2
        frames = pad_signal[indicies.astype(np.int32, copy=False)]
        window = np.tile(window_function(frame_length), (frame_number, 1))

    return  frames * window

def power_spectrum(frames, FFT_size=512):
    """
    Calculating power spectrum of each frame.

    Paramters:
    frames: array of frames. Each rowe is a frame.
    FFT_size: the lenght of FFT
    Returns: if frames are samples by frame the return will be (FFT size/2+1) by frames;
    """
    if np.shape(frames)[1] > FFT_size:
        logging.warn(
            'Frame length (%d) is bigger than FFT size (%d), data will be lost.',
            np.shape(frames)[1], FFT_size)
    return (((np.abs(np.fft.rfft(frames, FFT_size))) ** 2) / FFT_size)

def get_filterbanks(filter_number=20, fft_size=512, sample_rate=16000, low_freq=0,
                    high_freq=None):
    """
    creating filter bank with bandwidt of low_freq to high_ferq. Filters are stored in rows, columnt coressponds to
    FFT size.

    Paramters:
    filter_number: number of filters
    fft_size: FFT size
    sample_rate: sample rate of a signal that the filter bank will be applied
    low_freq: lower limit of filter bank in Hz
    high_freq: higher limit of filter bank in Hz
    returns: array of filter_number by (fft_size / 2 + 1). each row holds one filter.
    """
    high_freq = high_freq or sample_rate / 2

    # converting Hz to mels
    min_mel = freq_to_mel(low_freq)
    max_mel = freq_to_mel(high_freq)

    # spacing filter points on the mel scale
    mel_filter_points = np.linspace(min_mel, max_mel, filter_number + 2)

    
    # creating empty numpy array for filter bank
    filter_bank = np.zeros((filter_number, int(np.floor((fft_size / 2) + 1))))

    # computing values for previously created array
    bin = np.floor((fft_size + 1) * mel_to_freq(mel_filter_points) / sample_rate)
    for i in range(1, filter_number + 1):
        mel_filter_left = int(bin[i - 1])
        mel_filter = int(bin[i])
        mel_filter_right = int(bin[i + 1])

        for k in range(mel_filter_left, mel_filter):
            filter_bank[i - 1, k] = (k - bin[i - 1]) / (bin[i] - bin[i - 1])
        for k in range(mel_filter, mel_filter_right):
            filter_bank[i - 1, k] = (bin[i + 1] - k) / (bin[i + 1] - bin[i])

    return filter_bank

def cepstral_liftering(mfcc_cefficients, d=22):
    """
    Performing cepstral liftering on MFCC coefficients. Increasing SNR.

    Paramters:
    mfcc_cefficients: matric of mel cepstra. shape [number of frames by cepstral number]
    l: an L-th order fnite impulse response (FIR). 0 disables filter
    """
    if 0 < d:
        number_frames, mfcc_number = np.shape(mfcc_cefficients)
        i = np.arange(mfcc_number)
        lift = 1 + (d / 2.) * np.sin(np.pi * i / d)
        return lift * mfcc_cefficients
    else:
        return mfcc_cefficients
"""
#   removing silences from an wav file
def remove_silences(signal, resolution=100, frame_length=0.01, minimum_power=0.001, sample_rate=44100):
    duration = len(signal) / sample_rate # in seconds
    iterations = int(duration * resolution)
    step = int(sample_rate / resolution)
    window_length = np.floor(sample_rate * frame_length)
    signal_power = np.square(signal) / window_length #Normalized power to window duration

    start = np.array([])
    stop = np.array([])
    is_started = False

    for n in range(iterations):
        power = 10 * np.sum(signal_power[n * step : int(n * step + window_length)]) # sensitive
        if not is_started and power > minimum_power:
            start = np.append(start, n * step + window_length/2)
            is_started = True
        elif is_started and (power <= minimum_power or n == iterations-1):
            stop = np.append(stop, n * step + window_length/2)
            is_started = False

    if start.size == 0:
        start = np.append(start, 0)
        stop = np.append(stop, len(signal))

    start = start.astype(int)
    stop = stop.astype(int)

    # We don't want to eliminate EVERYTHING that's unnecessary
    # There should be a little boundary...
    # 200 frame buffer before and after

    # minus = ?
    if start[0] > 200:
        minus = 200
    else:
        minus = start[0]

    # plus = ?
    if (len(signal) - stop[0]) > 200:
        plus = 200
    else:
        plus = len(signal) - stop[0]

    signal_silence = np.empty([stop[0] - start[0]])

    for i in range(start[0] , stop[0]):
       signal_silence[i - start[0]] = signal[i]

    return signal_silence
"""
