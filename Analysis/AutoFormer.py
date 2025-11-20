import os

os.chdir("..")


import sys
import os
from pathlib import Path

# Load Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.GetCombinedData import getCombinedData

"""

State = 'FL'

GeorgiaCombinedData = getCombinedData(state=State)

GeorgiaCombinedData  = GeorgiaCombinedData.set_index('claims', append=True) 
#'Deaths', 'Recovered', 'Active'
GeorgiaCombinedData.columns
GeorgiaCombinedDataDeath = getCombinedData(state=State,column='Deaths')

GeorgiaCombinedDataDeath = GeorgiaCombinedDataDeath.set_index('claims', append=True) 


GeorgiaCombinedDataRecovered = getCombinedData(state=State, column='Recovered')


GeorgiaCombinedDataRecovered = GeorgiaCombinedDataRecovered.set_index('claims', append=True) 


GeorgiaCombinedDataActive = getCombinedData(state=State, column='Active')

GeorgiaCombinedDataActive = GeorgiaCombinedDataActive.set_index('claims', append=True) 



FinalGeorgiaCombinedData = GeorgiaCombinedData.join([GeorgiaCombinedDataDeath], how='inner')

FinalGeorgiaCombinedData.head()



FinalGeorgiaCombinedData = FinalGeorgiaCombinedData.reset_index()

FinalGeorgiaCombinedData.columns = ['Date', 'claims', 'Confirmed', 'Deaths']

FinalGeorgiaCombinedData = FinalGeorgiaCombinedData.set_index('Date', append=True) 

FinalGeorgiaCombinedData.head()

"""
# Save to CSV

# If the csv exists, skip the following line

#FinalGeorgiaCombinedData.to_csv("FinalFloridaCombinedData.csv", index=True)

FinalFloridaCombinedData = pd.read_csv("FinalFloridaCombinedData.csv")

from neuralforecast.models import TCN

from neuralforecast.models import autoformer





df = FinalFloridaCombinedData.copy()
df['ds'] = df.index
df['unique_id'] = 'series_1'
df = df.rename(columns=lambda x: x.strip())  

df = df.drop(columns=['Unnamed: 0'])  


df['ds'] = pd.to_datetime(df['Date'])
df = df.drop(columns=['Date'])
df['ds'] = pd.to_datetime(df['ds'].dt.date)


df['ds'] = pd.to_datetime(df['ds']) + pd.to_timedelta(1, unit='D')



df = df.rename(columns={ 'claims': 'y'})

"""
# Split the data into training, validation and test set

FinalFloridaCombinedData.shape

n = len(FinalFloridaCombinedData) 
train_size = int(n * 0.70)        # 70%
val_size   = int(n * 0.15)        # 15%
test_size  = n - train_size - val_size


train = FinalFloridaCombinedData.iloc[:train_size]
val   = FinalFloridaCombinedData.iloc[train_size : train_size + val_size]
test  = FinalFloridaCombinedData.iloc[train_size + val_size :]


print("Train:", train.shape)
print("Val:  ", val.shape)
print("Test: ", test.shape)

help(TCN)

"""

futr_cols = ['Deaths', 'Confirmed']

modelTCN1 = TCN(
    h=1,                      
    input_size=36,            
    inference_input_size=-1,  
    kernel_size=3,
    dilations=[1, 2, 4, 8, 16],

    # Encoder
    encoder_hidden_size=500,  
    encoder_activation='ReLU',

    # Decoder
    context_size=10,
    decoder_hidden_size=500,  
    decoder_layers=2,

    # Exogenous variables
   
    hist_exog_list=futr_cols,
    stat_exog_list=None
)

n = len(df)
train = df.iloc[:int(n*0.7)]
val   = df.iloc[int(n*0.7):int(n*0.85)]
test  = df.iloc[int(n*0.85):]

train['ds'].diff().value_counts()

val['ds'].diff().value_counts()

print("Train", train.shape)
print("Val ", val.shape)
print("Test ", test.shape)

from neuralforecast import NeuralForecast

nf = NeuralForecast(
    models=[modelTCN1],
    freq='W', 
    
)
nf.freq

val.head()



nf.fit(df=train)

val_pred = nf.predict(df=val)
val_pred

import neuralforecast
print(neuralforecast.__version__)

import utilsforecast
utilsforecast.__version__


val_predictions = []

for i in range(len(val)):
   
    df_until_now = pd.concat([train, val.iloc[:i]], axis=0)

   
    pred = nf.predict(df=df_until_now)
    val_predictions.append(pred)


len(val_predictions)
help(NeuralForecast)

# Package Versions Used
# neuralforecast: 1.7.4
# utilsforecast: 0.2.14
# Python: 3.10.19
