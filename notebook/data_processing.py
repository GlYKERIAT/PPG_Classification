import numpy as np
import pandas as pd 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import signal

def read_data(path= '../data/ppg.csv', selection = 'last' ):
    original_df = pd.read_csv(path)
    df = original_df.copy()
    df['CREATE_DATETIME'] = pd.to_datetime(df['CREATE_DATETIME'])   # Transform to datetime

    df = df[df["READING_CATEGORY"] == "PPG"]                        # Keeping only the PPG signals 

    encoder = LabelEncoder()
    df['PATIENT_CODE'] = encoder.fit_transform(df['PATIENT_CODE'])  # Label encoding the IDs

    df.drop(columns=["ID","READING_VALUE","READING_CATEGORY"],inplace=True)
    df.rename(columns={"CREATE_DATETIME":"DATE"},inplace=True)

    print("Nuber of unique patients:",len(df["PATIENT_CODE"].unique()))
    print("Unique years of birth:",df["YEAR_OF_BIRTH"].unique())
    new_df = select_duplicates(df, select = selection)
    return new_df

def select_duplicates(df, select= None):
    print("Amount of duplicate values: \n",df.duplicated(subset=['DATE', 'PATIENT_CODE'], keep=False).value_counts())
    if select=='last':
        duplicate = df[df.duplicated(subset=['DATE', 'PATIENT_CODE'], keep= 'last')]
        print("\nlen of duplicates which removed: \n",len(duplicate))
        df_unique = df.drop_duplicates(subset=['DATE', 'PATIENT_CODE'], keep='last')    #remove duplicates keep the last one only
        print("Amount of duplicate values after droping: \n",df_unique.duplicated(subset=['DATE', 'PATIENT_CODE'], keep=False).value_counts())       
        return df_unique
    
    if select=='first':
        duplicate = df[df.duplicated(subset=['DATE', 'PATIENT_CODE'], keep= 'first')]
        print("\nlen of duplicates which removed: \n",len(duplicate))
        df_unique = df.drop_duplicates(subset=['DATE', 'PATIENT_CODE'], keep='first')    #remove duplicates keep the last one only
        print("Amount of duplicate values after droping: \n",df_unique.duplicated(subset=['DATE', 'PATIENT_CODE'], keep=False).value_counts())       
        return df_unique
    
    if select== 'deleteAll':
        duplicate = df[df.duplicated(subset=['DATE', 'PATIENT_CODE'], keep= False)]
        print("\nlen of duplicates which removed: \n",len(duplicate))
        df_unique = df.drop_duplicates(subset=['DATE', 'PATIENT_CODE'], keep= False)    #remove duplicates keep the last one only
        print("Amount of duplicate values after droping: \n",df_unique.duplicated(subset=['DATE', 'PATIENT_CODE'], keep=False).value_counts())       
        return df_unique
    
    else:
        print("\nNo changes in duplicates")
        return df
    

def split_continuous_timeseries(df, min_duration=10):

    diff = df['DATE'].diff()
    continuous_dates = diff.dt.total_seconds().fillna(0) <= 1

    subdataframes = []
    for i, continuous in enumerate(continuous_dates):
        if i == 0 or not continuous:
            subdataframes.append(df.iloc[i:i+1])
        else:
            subdataframes[-1] = pd.concat([subdataframes[-1], df.iloc[i:i+1]])

    # Φιλτράρισμα χρονοσειρών με βάση τη διάρκεια
    filtered_subdfs = [subdf.reset_index(drop=True) for subdf in subdataframes if subdf.shape[0] >= min_duration]
    return filtered_subdfs

def preprocess_ppg(ppg_values):

    time_series = []
    for seconds_values in ppg_values:
        split_data = seconds_values[1:-1].split(', ')
        integer_list = [int(item) for item in split_data]
        time_series.extend(integer_list)
    return time_series





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