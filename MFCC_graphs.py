# Author: Kacper Wojtasik 2020

import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def _graph_windows(frame, windowed_frame):
    plt.plot(frame, label='No window applied')
    plt.plot(windowed_frame, label='Hamming window applied')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Frame vs Windowed Frame')
    plt.show()
    pass


def _graph_window_vs_original_frame(original, windowed):
    plt.plot(original, label='Hamming')
    plt.plot(windowed, label='Hanning')
    plt.title('Original vs Windowed frame')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    pass
def _graph_frequency_response(signal):
    fft_signal = abs(np.fft.fft(signal))
    plt.plot(fft_signal)
    plt.title('Frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.show()
    pass

def _graph_frequency_responses(signal, signal1):
    fft_signal = abs(np.fft.fft(signal))
    plt.plot(fft_signal, label='Original')
    plt.plot(abs(np.fft.fft(signal1)), label='Pre-emphaised')
    plt.title('Frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    pass

def _graph_signal_in_time_domain(signal, sample_rate=None):

    plt.title('Signal in time domain')

    if sample_rate != None:
        plt.plot(np.linspace(0, len(signal) / sample_rate, num=len(signal)),signal)
        plt.xlabel('Time [s]')
    else:
        plt.plot(signal)
        plt.xlabel('Samples')

    plt.ylabel('Amplitude ')
    plt.show()
    pass

def _graph_framed_signal_in_time_domain(frames):
    signal = np.zeros([1])
    for i in range(frames.shape[0]):
        signal = np.append(signal, frames[i])
    plt.title('Signal in time domain')
    plt.plot(signal)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude ')
    plt.show()
    pass

def _graph_framed_frequency_response(frames):
    signal = np.zeros([1])
    for i in range(frames.shape[0]):
        signal = np.append(signal, frames[i])
    fft_signal = abs(np.fft.fft(signal))
    plt.plot(fft_signal)
    plt.title('Frequency response')
    plt.xlabel('Bandwidth [not Hz]')
    plt.ylabel('Amplitude')
    plt.show()
    pass

def _graph_power_spectrum_power_frames(frames, sample_rate=16000):
    plt.imshow(frames.T, aspect='auto', origin='lower')
    #plt.colorbar()
    plt.title('Power Spectrum')
    plt.ylabel('FFT bins (Frequncy 0-' + str(int(sample_rate/2)) + "Hz)")
    plt.xlabel('Number of frame')
    plt.show()
    pass

def _graph_mel_filter_banks(filter_banks, sample_rate=16000):
    for i in range(filter_banks.shape[0]):
        plt.plot(filter_banks[i])
    plt.title("Filter banks")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency in range 0Hz to " + str(sample_rate / 2) + "Hz")
    plt.show()
    pass

def _graph_features(features):
    plt.title('Features of signal')
    plt.imshow(features.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('Number of frames (time)')
    plt.ylabel('Filter number')
    plt.show()
    pass

def _graph_mfcc(mfcc):
    plt.title("MFCC coefficients")
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.colorbar()
    plt.ylabel('Number of frames (time)')
    plt.xlabel('MFCC coefficients')
    plt.show()
    pass

def _graph_two_mel_filter_banks_(filter_banks, filter_banks1, sample_rate=16000):
    plt.subplot(1,2,1)
    for i in range(filter_banks.shape[0]):
        plt.plot(filter_banks[i])
    plt.title("Filter banks")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency in range 0Hz to " + str(sample_rate / 2) + "Hz")
    plt.subplot(1,2,2)
    for i in range(filter_banks1.shape[0]):
        plt.plot(filter_banks1[i])
    plt.title("Filter banks")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency in range 0Hz to " + str(sample_rate / 2) + "Hz")
    plt.show()
    pass

def _graph_filter_bank_power(filter_bank):
    plt.imshow(filter_bank.T ,origin='lower', aspect='auto')
    plt.xlabel('Number of frame (time)')
    plt.title('Filter bank power spectrum in log')
    plt.ylabel('Filter number (Frequency)')
    plt.show()
    pass