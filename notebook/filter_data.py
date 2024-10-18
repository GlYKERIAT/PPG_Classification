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

def bubble_entropy(time_series, r, m):
    """
    Calculate Bubble Entropy of a time series.

    Parameters:
    - time_series: 1D array-like of the time-series data
    - m: Embedding dimension (usually 2)
    - r: Similarity tolerance (usually between 0.1 and 0.25 times the standard deviation of the time-series)

    Returns:
    - bubble_entropy_value: The calculated Bubble Entropy value
    """

    N = len(time_series)
    
    # Normalize the time series
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # Step 1: Create the embedded matrix
    embedded_matrix = np.array([time_series[i:N-m+1+i] for i in range(m)])
    
    # Step 2: Count the number of 'bubbles' in the embedded matrix
    def count_bubbles(matrix, threshold):
        count = 0
        for i in range(matrix.shape[1]):
            for j in range(i+1, matrix.shape[1]):
                if np.linalg.norm(matrix[:, i] - matrix[:, j], ord=np.inf) > threshold:
                    count += 1
        return count
    
    # Step 3: Calculate bubble entropy for embedding dimensions m and m+1
    Cm = count_bubbles(embedded_matrix, r)
    
    # Embed for dimension m+1
    embedded_matrix_m1 = np.array([time_series[i:N-(m+1)+1+i] for i in range(m+1)])
    Cm1 = count_bubbles(embedded_matrix_m1, r)
    
    # Step 4: Compute the Bubble Entropy value
    if Cm > 0 and Cm1 > 0:
        bubble_entropy_value = -np.log(Cm1 / Cm)
    else:
        bubble_entropy_value = np.inf  # Undefined if no bubbles found
    
    return bubble_entropy_value



def calculate_crossings(ppg_signal):
    zero_crossing_indices = np.nonzero(np.diff(np.array(ppg_signal) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(ppg_signal) > np.nanmean(ppg_signal)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(ppg_signal, r_m):
    entropy = calculate_entropy(ppg_signal)
    crossings = calculate_crossings(ppg_signal)
    statistics = calculate_statistics(ppg_signal)

    bubble_entropy_list = []
    #print("edww")
    for i in range(2,5,1):
        #print("i", i)
        for j in np.arange(0.1,0.26,0.05):
            
            #print("j",j)
            bubble_entropy_list.append(bubble_entropy(ppg_signal, m= i, r= j))
                           
    # r = r_m["r"]
    # m = r_m["m"]
    #print("r", r)
    #print("m", m)
    #bubble_value = bubble_entropy(ppg_signal, m=m , r=r)
    #features = [entropy] + crossings + statistics + bubble_entropy_list
    #features = [entropy] + crossings + statistics + [bubble_value] 
    #features = [entropy] + crossings + statistics
    #print(len(features))
    return bubble_entropy_list
    #return features



def extract_features( ppg_signal, features_params, filter_params, bubble_params):
    method = features_params['method']
    r_m = bubble_params
    # # 1) Filtering the Signal
    if(method == 'simple'):
        ppg_signal_filtered = apply_lowpass_filter(ppg_signal, filter_params)
        features = get_features(ppg_signal_filtered, r_m)

    # # 2) Doing the Wavelet analysis to extract the detailed and approximate coefficients (this is considered a filtering method as well)
    if(method == 'wavelet'):
        ppg_signal_filtered = apply_lowpass_filter(ppg_signal, filter_params)        ##wavelet with filtered signal all posible 
        #print(features_params)
        wavelet_type = features_params['type']
        level = features_params['level']  
              
        coeffs = wavedec(ppg_signal_filtered, wavelet_type, level=level) #### να αλλάξω το επίπεδο σε 2 και 3 και 4 να δούμε ότι το 5 το καλύτερο
        # The first element is the approximation coefficient (cA), and the rest are detailed coefficients (cDs)
        cA = coeffs[0]  # Approximate coefficient at the highest level
        cDs = coeffs[1:]  # Detailed coefficients

        # Get the coefficient names based on the level
        coeff_names = [f'cD{i}' for i in range(level, 0, -1)]
        coeff_names.insert(0, f'cA{level}')  # Add the approximation coefficient at the highest level
        
        # Extracting features from the approximation coefficient
        features = get_features(cA, r_m)

        # Loop through the detailed coefficients dynamically based on the wavelet level
        for i in range(len(cDs)):
            features += get_features(cDs[i], r_m)
    #features = get_features(ppg_signal)
    #features = features_6 
    print(len(features))
    return features, coeff_names

import numpy as np
