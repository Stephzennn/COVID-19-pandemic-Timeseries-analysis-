from neuralforecast.auto import AutoAutoformer

# Retrofit
import os
os.chdir("..")
from neuralforecast import NeuralForecast
from neuralforecast.models import Autoformer
from neuralforecast.losses.pytorch import MAE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
def evaluate_performance(true, pred, model_name="Model"):
    mspe = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-6)))
    pm = (np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2))

    print(f"=== {model_name} Performance ===")
    print(f"MSPE: {mspe:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"PM:   {pm:.4f}")
    print("-" * 40)
    return mspe, mae, mape, pm

FinalFloridaCombinedData = pd.read_csv("Final_GA_CombinedData.csv")


df = FinalFloridaCombinedData.copy()
df['ds'] = df.index
df['unique_id'] = 'series_1'
df = df.rename(columns=lambda x: x.strip())  
try:
    df = df.drop(columns=['Unnamed: 0'])  
except:
    x =1 

df['ds'] = pd.to_datetime(df['Date'])
df = df.drop(columns=['Date'])
df['ds'] = pd.to_datetime(df['ds'].dt.date)

df['ds'] = pd.to_datetime(df['ds']) + pd.to_timedelta(1, unit='D')

df = df.rename(columns={'claims': 'y'})

df = df.drop(columns=['Active'])

futr_cols = ['Deaths', 'Confirmed']  


n = len(df)
train = df.iloc[:int(n*0.7)]
val   = df.iloc[int(n*0.7):int(n*0.85)]
test  = df.iloc[int(n*0.85):]

print("Train:", train.shape)
print("Val:",   val.shape)
print("Test:",  test.shape)

train.head()


config = dict(
    input_size=7,
    hist_exog_list=[],
    stat_exog_list=[],
    futr_exog_list=[]
)


model = AutoAutoformer(h=1, loss=MAE(), config=config)

nf = NeuralForecast(models=[model], freq='W')

nf.fit(train)

forecast = nf.predict()