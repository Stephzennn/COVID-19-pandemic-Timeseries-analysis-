
# Package Versions Used
# neuralforecast: 1.7.4
# utilsforecast: 0.2.14
# Python: 3.10.19



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

FinalFloridaCombinedData = pd.read_csv("FinalFloridaCombinedData.csv")


df = FinalFloridaCombinedData.copy()
df['ds'] = df.index
df['unique_id'] = 'series_1'
df = df.rename(columns=lambda x: x.strip())  
df = df.drop(columns=['Unnamed: 0'])  

df['ds'] = pd.to_datetime(df['Date'])
df = df.drop(columns=['Date'])
df['ds'] = pd.to_datetime(df['ds'].dt.date)

# Sometimes the first day is excluded; shift +1 day
df['ds'] = pd.to_datetime(df['ds']) + pd.to_timedelta(1, unit='D')

# Target column for NeuralForecast:
df = df.rename(columns={'claims': 'y'})

# FUTURE EXOGENOUS VARIABLES
futr_cols = ['Deaths', 'Confirmed']  


n = len(df)
train = df.iloc[:int(n*0.7)]
val   = df.iloc[int(n*0.7):int(n*0.85)]
test  = df.iloc[int(n*0.85):]

print("Train:", train.shape)
print("Val:",   val.shape)
print("Test:",  test.shape)


model = Autoformer(
    h=1,                      
    input_size=36,             
    hidden_size=16,
    conv_hidden_size=32,
    n_head=2,

    loss=MAE(),

    futr_exog_list=None,   
    stat_exog_list=None,

    scaler_type='robust',
    learning_rate=1e-3,
    max_steps=300,
    val_check_steps=50,
    early_stop_patience_steps=2
)


nf = NeuralForecast(
    models=[model],
    freq='W'
)

nf.fit(df=train, val_size=len(val))

val_pred = nf.predict(df=val)

val_pred.head()

rolling_preds = []

for i in range(val.shape[0]):
    # data available up to this point
    df_until_now = pd.concat([train, val.iloc[:i]], axis=0)

    # forecast 1 step ahead
    pred_i = nf.predict(df=df_until_now)

    # store prediction (align with actual val timestamp)
    pred_i['ds'] = val.iloc[i]['ds']
    rolling_preds.append(pred_i)

rolling_preds = pd.concat(rolling_preds).reset_index(drop=True)

rolling_preds

val

val_plot = val.merge(
    rolling_preds[['ds', 'Autoformer']],
    on='ds',
    how='left'
)

val_plot

evaluate_performance(val_plot['y'],val_plot['Autoformer'], model_name="AutoFormer")

plt.figure(figsize=(12,6))
plt.plot(val_plot['ds'], val_plot['y'], label="True (Val)", c="black")
plt.plot(val_plot['ds'], val_plot['Autoformer'], label="Rolling Forecast", c="blue")
plt.grid()
plt.legend()
plt.show()


# Hyper parameter Check 

"""
hidden_sizes = [8, 16, 32, 64, 128]
conv_hidden_sizes = [16, 32, 64, 128, 256]
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
n_heads = [1, 2, 4, 8, 16]
encoder_layers = [1, 2, 3, 4, 6]
decoder_layers = [1, 2, 3, 4, 6]
"""
hidden_sizes      = [16, 64]
conv_hidden_sizes = [32, 128]
learning_rates    = [1e-3, 1e-4]
n_heads           = [2, 4]
encoder_layers    = [2, 4]
decoder_layers    = [2, 4]

top_models = [] 



for hidden_size in hidden_sizes:
    for conv_hidden_size in conv_hidden_sizes:
        for lr in learning_rates:
            for n_head in n_heads:
                for enc_layers in encoder_layers:
                    for dec_layers in decoder_layers:
                        model = Autoformer(
                            h=1,                      
                            input_size=36,             
                            hidden_size=hidden_size,
                            conv_hidden_size=conv_hidden_size,
                            n_head=n_head,
                            encoder_layers=enc_layers,
                            decoder_layers=dec_layers,
                            loss=MAE(),
                            
                            futr_exog_list=None,   
                            stat_exog_list=None,

                            scaler_type='robust',
                            learning_rate=lr,
                            max_steps=200,
                            val_check_steps=50,
                            early_stop_patience_steps=2
                                )


                        nf = NeuralForecast(
                            models=[model],
                            freq='W'
                        )

                        nf.fit(df=train, val_size=len(val))

                        val_pred = nf.predict(df=val)

                        val_pred.head()

                        rolling_preds = []

                        for i in range(val.shape[0]):
                            # data available up to this point
                            df_until_now = pd.concat([train, val.iloc[:i]], axis=0)

                            # forecast 1 step ahead
                            pred_i = nf.predict(df=df_until_now)

                            # store prediction (align with actual val timestamp)
                            pred_i['ds'] = val.iloc[i]['ds']
                            rolling_preds.append(pred_i)

                        rolling_preds = pd.concat(rolling_preds).reset_index(drop=True)


                        val_plot = val.merge(
                            rolling_preds[['ds', 'Autoformer']],
                            on='ds',
                            how='left'
                        )

                        val_plot

                        model_name = (
                            f"AF_h{hidden_size}"
                            f"_c{conv_hidden_size}"
                            f"_lr{lr}"
                            f"_hd{n_head}"
                            f"_enc{enc_layers}"
                            f"_dec{dec_layers}"
                        )

                        #evaluate_performance(val_plot['y'],val_plot['Autoformer'], model_name="AutoFormer")
                        mspe, mae, mape, pm = evaluate_performance(
                            val_plot['y'],
                            val_plot['Autoformer'],
                            model_name= model_name
                        )

                        # ---------------------------
                        # SAVE ONLY TOP 3 MODELS
                        # ---------------------------
                        result = {
                            'hidden_size': hidden_size,
                            'conv_hidden_size': conv_hidden_size,
                            'lr': lr,
                            'n_head': n_head,
                            'mspe': mspe,
                            'mae': mae,
                            'mape': mape,
                            'pm': pm
                        }

                        top_models.append(result)

                        
# Sort by MAE and keep only top 3
top_models = sorted(top_models, key=lambda x: x['mae'])[:3]
print("\n=== TOP 3 AUTOFORMER MODELS BY MAE ===")

for rank, model in enumerate(top_models, start=1):
    print(f"\n# {rank}")
    for k, v in model.items():
        print(f"{k}: {v}")