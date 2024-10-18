# %% [markdown]
# Read and filtered Data

# %%
import numpy as np
import pandas as pd 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import signal

def apply_lowpass_filter(ppg_signal, filter_params):

    filter_type = filter_params['type']
    sampling_rate = filter_params['sampling_rate']
    cutoff = filter_params['cutoff']

    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff / nyquist_freq
    
    if filter_type == 'None':
       return ppg_signal 
    if filter_type == 'butter':
        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    elif filter_type == 'cheby1':
        b, a = signal.cheby1(4, 0.5, normalized_cutoff, btype='low', analog=False)
    elif filter_type == 'cheby2':
        b, a = signal.cheby2(4, 20, normalized_cutoff, btype='low', analog=False)
    elif filter_type == 'elliptic':
        b, a = signal.ellip(4, 0.5, 20, normalized_cutoff, btype='low', analog=False)
    else:
        raise ValueError("Invalid filter type. Choose 'butter', 'cheby1', 'cheby2', or 'elliptic'.")
    
    ppg_filtered = signal.filtfilt(b, a, ppg_signal)
    return ppg_filtered

import scipy.special
from scipy.signal import find_peaks, welch
from collections import Counter
from pywt import wavedec

def calculate_entropy(ppg_signal):
    counter_values = Counter(ppg_signal).most_common()
    probabilities = [elem[1]/len(ppg_signal) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(ppg_signal):
    n5 = np.nanpercentile(ppg_signal, 5)
    n25 = np.nanpercentile(ppg_signal, 25)
    n75 = np.nanpercentile(ppg_signal, 75)
    n95 = np.nanpercentile(ppg_signal, 95)
    median = np.nanpercentile(ppg_signal, 50)
    mean = np.nanmean(ppg_signal)
    std = np.nanstd(ppg_signal)
    var = np.nanvar(ppg_signal)
    rms = np.nanmean(np.power(np.power(ppg_signal, 2), 0.5))
    skew = pd.Series(ppg_signal).skew()
    kurtosis = pd.Series(ppg_signal).kurt()
    min = np.min(ppg_signal)
    max = np.max(ppg_signal)

    # Heart rate related features
    # sampling_rate = 50
    # peaks, _ = find_peaks(ppg_signal, height=0)                         # Find peaks in PPG signal
    # instant_hr = 60 * len(peaks) / len(ppg_signal) * sampling_rate      # Instantaneous Heart Rate
    # rri = np.diff(peaks) / sampling_rate                                # RR intervals
    # hrv = np.std(rri)                                                   # Heart Rate Variability

    # # Morphological features
    # peaks, _ = find_peaks(ppg_signal, distance=50)
    # peak_count = len(peaks)
    # peak_mean = np.mean(ppg_signal[peaks]) if len(peaks) > 0 else 0
    # peak_std = np.std(ppg_signal[peaks]) if len(peaks) > 0 else 0

    # Frequency domain features
    # freqs, psd = welch(ppg_signal)
    # psd_mean = np.mean(psd)
    # psd_std = np.std(psd)
    # psd_peak = freqs[np.argmax(psd)]

    # Time domain features
    # diff_signal = np.diff(ppg_signal)
    # diff_mean = np.mean(diff_signal)
    # diff_std = np.std(diff_signal)

    return [n5, n25, n75, n95, median, mean, std, var, rms, skew, kurtosis, min, max]

def calculate_crossings(ppg_signal):
    zero_crossing_indices = np.nonzero(np.diff(np.array(ppg_signal) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(ppg_signal) > np.nanmean(ppg_signal)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(ppg_signal):
    entropy = calculate_entropy(ppg_signal)
    crossings = calculate_crossings(ppg_signal)
    statistics = calculate_statistics(ppg_signal)
    return [entropy] + crossings + statistics


def extract_features(ppg_signal, wavelet_level, features_params, filter_params):
    method = features_params['method']
    
    # 1) Filtering the Signal
    if method == 'simple':
        ppg_signal_filtered = apply_lowpass_filter(ppg_signal, filter_params)
        features = get_features(ppg_signal_filtered)

    # 2) Doing the Wavelet analysis to extract the detailed and approximate coefficients
    if method == 'wavelet':
        ppg_signal_filtered = apply_lowpass_filter(ppg_signal, filter_params)
        wavelet_type = features_params['type']
        level = wavelet_level  # Dynamically set the wavelet level based on the input
        coeffs = wavedec(ppg_signal_filtered, wavelet_type, level=level)
        
        # Extract the approximate and detailed coefficients based on the level
        cA = coeffs[0]  # Approximate coefficient at the highest level
        cDs = coeffs[1:]  # Detailed coefficients at each level
        
        # Extracting features from all the coefficients
        features = []
        features.append(get_features(cA))  # Features from the approximate coefficients

        # Loop through the detailed coefficients dynamically based on the wavelet level
        for i in range(len(cDs)):
            features.append(get_features(cDs[i]))

        # Flatten the list of features into a single list
        features = sum(features, [])

    return features

