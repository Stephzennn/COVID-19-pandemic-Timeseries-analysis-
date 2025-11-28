
# Package Versions Used
# neuralforecast: 1.7.4
# utilsforecast: 0.2.14
# Python: 3.10.19

#We Did a grid search using COLAB T4 GPU


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

# Sometimes the first day is excluded; shift +1 day
df['ds'] = pd.to_datetime(df['ds']) + pd.to_timedelta(1, unit='D')

# Target column claims:
#df = df.rename(columns={'claims': 'y'})

# Target column 'Confirmed'
#df = df.rename(columns={'Confirmed': 'y'})

# Target column 'Deaths'
df = df.rename(columns={'Deaths': 'y'})


df = df.drop(columns=['Active'])
# FUTURE EXOGENOUS VARIABLES
futr_cols = ['Deaths', 'Confirmed']  


n = len(df)
train = df.iloc[:int(n*0.7)]
val   = df.iloc[int(n*0.7):int(n*0.85)]
test  = df.iloc[int(n*0.85):]

print("Train:", train.shape)
print("Val:",   val.shape)
print("Test:",  test.shape)

train.head()



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
                            early_stop_patience_steps=2,
                            enable_progress_bar=False,
                            enable_model_summary=False,

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

                        result = {
                            'hidden_size': hidden_size,
                            'conv_hidden_size': conv_hidden_size,
                            'lr': lr,
                            'n_head': n_head,
                            'enc_layers' :enc_layers,
                            'dec_layers' : dec_layers,
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



"""
=== TOP 3 AUTOFORMER MODELS BY MAE ===

# 1
hidden_size: 16
conv_hidden_size: 128
lr: 0.0001
n_head: 4
enc_layers: 4
dec_layers: 4
mspe: 1827463.5054851745
mae: 1087.5776685631793
mape: 0.17673068714031312
pm: 2.2589641080201557

# 2
hidden_size: 16
conv_hidden_size: 128
lr: 0.0001
n_head: 2
enc_layers: 4
dec_layers: 4
mspe: 1835138.4781292807
mae: 1094.8162470278533
mape: 0.17799333112702048
pm: 2.268451294867408

# 3
hidden_size: 64
conv_hidden_size: 128
lr: 0.0001
n_head: 2
enc_layers: 2
dec_layers: 2
mspe: 2176483.54390021
mae: 1216.295749830163
mape: 0.19870637896432275
pm: 2.6903947425542567


"""


# Predict on test set 

# Hyperparameters
hidden_size = 16
conv_hidden_size = 128
lr = 0.0001
n_head = 4
enc_layers = 4
dec_layers = 4

# Define Model
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
    
    #hist_exog_list = ,
    scaler_type='robust',
    learning_rate=lr,
    max_steps=200,
    val_check_steps=50,
    early_stop_patience_steps=2,
    enable_progress_bar=False,
    enable_model_summary=False,
)

nf = NeuralForecast(models=[model], freq='W')
nf.fit(df=train, val_size=len(val))  

rolling_preds = []

for i in range(test.shape[0]):
    hist_df = pd.concat([train, val, test.iloc[:i]], axis=0)

    pred_i = nf.predict(df=hist_df)
    pred_i['ds'] = test.iloc[i]['ds']  

    rolling_preds.append(pred_i)

rolling_preds = pd.concat(rolling_preds).reset_index(drop=True)

# Merge Predictions with TEST Truth
test_forecast = test.merge(
    rolling_preds[['ds', 'Autoformer']],
    on='ds',
    how='left'
)

model_name = (
    f"AF_h{hidden_size}"
    f"_c{conv_hidden_size}"
    f"_lr{lr}"
    f"_hd{n_head}"
    f"_enc{enc_layers}"
    f"_dec{dec_layers}"
)

# Evaluate ON TEST ONLY
mspe, mae, mape, pm = evaluate_performance(
    test_forecast['y'],
    test_forecast['Autoformer'],
    model_name=model_name
)

print(model_name)
print("MSPE:", mspe)
print("MAE:", mae)
print("MAPE:", mape)
print("PM:", pm)

test_forecast.head()


# ================= PLOT: LAST TRAIN + VAL + TEST =================
last_n_train = 30 

train_zoom = train.iloc[-last_n_train:]

plt.figure(figsize=(14, 6))

# Actual segments
plt.plot(train_zoom['ds'], train_zoom['y'], label='Train (Tail)', alpha=0.8)
plt.plot(val['ds'], val['y'], label='Validation', alpha=0.8)
plt.plot(test['ds'], test['y'], label='Test Actual', alpha=0.8)

# Forecast
plt.plot(test_forecast['ds'], test_forecast['Autoformer'],
         label='Forecast (Test)',
         linewidth=2)


# Confidence Intervals 
if 'Autoformer-lo-90' in rolling_preds.columns:
    merged_ci = test.merge(
        rolling_preds[['ds', 'Autoformer-lo-90', 'Autoformer-hi-90']],
        on='ds', how='left'
    )
    plt.fill_between(
        merged_ci['ds'],
        merged_ci['Autoformer-lo-90'],
        merged_ci['Autoformer-hi-90'],
        alpha=0.2, label='90% CI'
    )

plt.axvline(val['ds'].iloc[0], linestyle='--', color='gray', alpha=0.8)
plt.axvline(test['ds'].iloc[0], linestyle='--', color='gray', alpha=0.8)

plt.title(f'Train Tail + Val + Test Forecast\n{model_name}')
plt.xlabel('Date')
plt.ylabel('Claims')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
#FOR CONFIRMED CASES
# ================= PLOT: LAST TRAIN + VAL + TEST =================
last_n_train = 30
train_zoom = train.iloc[-last_n_train:]

plt.figure(figsize=(14, 6))

# Actual segments
plt.plot(train_zoom['ds'], train_zoom['y'], label='Train (Tail)', alpha=0.8)
plt.plot(val['ds'], val['y'], label='Validation', alpha=0.8)
plt.plot(test['ds'], test['y'], label='Test Actual', alpha=0.8)

# Forecast
plt.plot(test_forecast['ds'], test_forecast['Autoformer'],
         label='Forecast (Test)', linewidth=2)

# Confidence Intervals
if 'Autoformer-lo-90' in rolling_preds.columns:
    merged_ci = test.merge(
        rolling_preds[['ds', 'Autoformer-lo-90', 'Autoformer-hi-90']],
        on='ds', how='left'
    )
    plt.fill_between(
        merged_ci['ds'],
        merged_ci['Autoformer-lo-90'],
        merged_ci['Autoformer-hi-90'],
        alpha=0.2, label='90% CI'
    )

plt.axvline(val['ds'].iloc[0], linestyle='--', color='gray', alpha=0.8)
plt.axvline(test['ds'].iloc[0], linestyle='--', color='gray', alpha=0.8)

# Updated labels
plt.title(f'Confirmed Cases: Train Tail + Validation + Test Forecast\n{model_name}')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""


"""
# FOR DEATHS
# ================= PLOT: LAST TRAIN + VAL + TEST (Deaths) =================
last_n_train = 30
train_zoom = train.iloc[-last_n_train:]

plt.figure(figsize=(14, 6))

# Actual segments
plt.plot(train_zoom['ds'], train_zoom['y'], label='Train Deaths (Tail)', alpha=0.8)
plt.plot(val['ds'], val['y'], label='Validation Deaths', alpha=0.8)
plt.plot(test['ds'], test['y'], label='Test Actual Deaths', alpha=0.8)

# Forecast
plt.plot(test_forecast['ds'], test_forecast['Autoformer'],
         label='Forecasted Deaths (Test)', linewidth=2)

# Confidence Intervals
if 'Autoformer-lo-90' in rolling_preds.columns:
    merged_ci = test.merge(
        rolling_preds[['ds', 'Autoformer-lo-90', 'Autoformer-hi-90']],
        on='ds', how='left'
    )
    plt.fill_between(
        merged_ci['ds'],
        merged_ci['Autoformer-lo-90'],
        merged_ci['Autoformer-hi-90'],
        alpha=0.2, label='90% CI (Deaths)'
    )

plt.axvline(val['ds'].iloc[0], linestyle='--', color='gray', alpha=0.8)
plt.axvline(test['ds'].iloc[0], linestyle='--', color='gray', alpha=0.8)

# Updated labels for Deaths
plt.title(f'Deaths: Train Tail + Validation + Test Forecast\n{model_name}')
plt.xlabel('Date')
plt.ylabel('Deaths')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""