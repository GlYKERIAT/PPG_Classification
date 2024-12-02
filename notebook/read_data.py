#read and select duplicates

import numpy as np
import pandas as pd 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


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
    


