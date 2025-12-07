

import os

os.chdir("..")


import sys
import os
from pathlib import Path

# Load Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load Time Series Analysis Packages
from statsmodels.tsa.api import VAR
import warnings
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=RuntimeWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='statsmodels')

#from data.GetCombinedData import getCombinedData



FinalFloridaCombinedData = pd.read_csv("Final_FL_CombinedData.csv")

FinalFloridaCombinedData.head()

FinalFloridaCombinedData = FinalFloridaCombinedData.drop(columns=["Active"])

#FinalFloridaCombinedData = FinalFloridaCombinedData.set_index('Date', append=True) 




df = FinalFloridaCombinedData.copy()
"""
df['ds'] = df.index
df['unique_id'] = 'series_1'
df = df.rename(columns=lambda x: x.strip())  
try:
    df = df.drop(columns=['Unnamed: 0'])  
except:
    x =1 
"""
df['ds'] = pd.to_datetime(df['Date'])
df = df.drop(columns=['Date'])
df['ds'] = pd.to_datetime(df['ds'].dt.date)

df.head()



# Sometimes the first day is excluded; shift +1 day
df['ds'] = pd.to_datetime(df['ds']) + pd.to_timedelta(1, unit='D')

df = df.set_index('ds', append=True) 
# Target column claims:
#df = df.rename(columns={'claims': 'y'})

# Target column 'Confirmed'
#df = df.rename(columns={'Confirmed': 'y'})

# Target column 'Deaths'
#df = df.rename(columns={'Deaths': 'y'})


df = df.drop(columns=['Recovered'])
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
varModelSelect = VAR(train).select_order(maxlags=10)
print(varModelSelect.summary())

varModel = VAR(train).fit(maxlags=6)
print(varModel.summary())

def create_lag_equation(tsData, maxLag, colName):
    rhs = [colName] + [f"{colName}_l{lag}" for lag in range(1, maxLag+1)]
    system = pd.concat([tsData.shift(i) for i in range(maxLag+1)], axis=1).dropna()
    system.columns = rhs
    return {"system": system, "terms": rhs[1:]}

def step(data, steps, maxLag, colName, verbose=False):
    system, terms = construct_data(data, maxLag)
    for _ in range(steps):
        conti, terms = step_backward(system, terms, colName, verbose)
        if not terms or not conti:
            break
    return terms + ['(Intercept)']

def construct_data(data, maxLag):
    dataList = []
    terms = []
    for col in data.columns:
        system = create_lag_equation(data[col], maxLag=maxLag, colName=col)
        dataList.append(system['system'])
        terms.extend(system['terms'])
    return pd.concat(dataList, axis=1).dropna(), terms

def step_backward(system, terms, colName, verbose=False):
    aic_full, rss_full, _ = step_backward_helper(system, terms, colName, 'none')
    aicDict = {'term': ['none'], 'DF': [0], 'SumSq': [0], "RSS": [rss_full], 'AIC': [aic_full]}
    for aTerm in terms:
        aic, rss, df = step_backward_helper(system, terms, colName, aTerm)
        aicDict['term'].append(aTerm)
        aicDict['AIC'].append(aic)
        aicDict['DF'].append(df)
        aicDict['SumSq'].append(rss - rss_full)
        aicDict['RSS'].append(rss)
    aicDF = pd.DataFrame(aicDict)
    if verbose:
        print(f"Step: AIC={aic_full:.4f}")
        print(f"{colName} ~ {' + '.join(terms)}")
        print(aicDF)
    leastImportantTerm = aicDF.loc[aicDF['AIC'].idxmin(), 'term']
    if leastImportantTerm == 'none':
        return False, terms
    return True, [term for term in terms if term != leastImportantTerm]

def step_backward_helper(system, terms, colName, excludeTerm):
    currTerms = [term for term in terms if term != excludeTerm and term != 'none']
    model = smf.ols(f"{colName} ~ {' + '.join(currTerms)}", data=system).fit()
    return model.aic, model.ssr, 1


### Stepwise Regression
coef_Claims = step(train, steps=5, maxLag=6, colName='claims') 
print(f"Claims Model {coef_Claims}")
coef_Covid = step(train, steps=5, maxLag=6, colName='Confirmed') 
print(f"Covid Model {coef_Covid}")


import numpy as np
from sklearn.metrics import mean_absolute_error

# === CLAIMS Stepwise-selected lags ===
claims_lags = ['claims_l1','claims_l2','claims_l3','claims_l4','claims_l5','claims_l6',
               'Confirmed_l1','Confirmed_l3','Confirmed_l5',
               'Deaths_l1','Deaths_l3','Deaths_l5','Deaths_l6']

# === CONFIRMED Stepwise-selected lags ===
confirmed_lags = ['claims_l2','claims_l3',
                  'Confirmed_l1','Confirmed_l2','Confirmed_l3','Confirmed_l4','Confirmed_l5','Confirmed_l6',
                  'Deaths_l1','Deaths_l2','Deaths_l4','Deaths_l5','Deaths_l6']



import numpy as np
from sklearn.metrics import mean_absolute_error

# Stepwise-selected terms (your final lists)
claims_terms = ['claims_l1','claims_l2','claims_l3','claims_l4','claims_l5','claims_l6',
                'Confirmed_l1','Confirmed_l3','Confirmed_l5',
                'Deaths_l1','Deaths_l3','Deaths_l5','Deaths_l6']

confirmed_terms = ['claims_l2','claims_l3',
                   'Confirmed_l1','Confirmed_l2','Confirmed_l3','Confirmed_l4','Confirmed_l5','Confirmed_l6',
                   'Deaths_l1','Deaths_l2','Deaths_l4','Deaths_l5','Deaths_l6']


# Generate restricted forecast for 1 step ahead
def restricted_forecast(model, history):
    k_ar = model.k_ar
    cols = history.columns
    params = model.params

    intercept = params.iloc[0].values
    lag_params = params.iloc[1:].values  # drop intercept row

    claims_pred = intercept[0]
    confirmed_pred = intercept[1]
    deaths_pred = intercept[2]  # full model for deaths

    for lag in range(1, k_ar+1):
        lag_row = lag_params[(lag-1)*len(cols):(lag)*len(cols)]
        for col_i, col in enumerate(cols):
            term = f"{col}_l{lag}"

            if term in claims_terms:
                claims_pred += lag_row[col_i] * history[col].iloc[-lag]

            if term in confirmed_terms:
                confirmed_pred += lag_row[col_i] * history[col].iloc[-lag]

    return np.array([claims_pred, confirmed_pred, deaths_pred])


def rolling_eval(model, history, future):
    preds_claims = []
    preds_confirmed = []

    for i in range(len(future)):
        pred = restricted_forecast(model, history)
        preds_claims.append(pred[0])
        preds_confirmed.append(pred[1])

        history = pd.concat([history, future.iloc[[i]]])  # update with actual future data

    return (
        mean_absolute_error(future["claims"], preds_claims),
        mean_absolute_error(future["Confirmed"], preds_confirmed)
    )


print("\nEvaluating Restricted VAR(6) Model...")

# Validation results
mae_claims_val, mae_confirmed_val = rolling_eval(varModel, train.copy(), val.copy())
print("Validation MAE:")
print(f"Claims:     {mae_claims_val:.3f}")
print(f"Confirmed:  {mae_confirmed_val:.3f}")

# Test results
mae_claims_test, mae_confirmed_test = rolling_eval(varModel, pd.concat([train, val]), test.copy())
print("\nTest MAE:")
print(f"Claims:     {mae_claims_test:.3f}")
print(f"Confirmed:  {mae_confirmed_test:.3f}")




###############################################
# ROLLING FORECAST FOR VAR MODEL (LAG = 2)
###############################################

lag_order = 2  # Based on our chosen lag

# Helper function for rolling expanding window predictions
def rolling_var_forecast(train_data, future_data, lag_order):
    preds = []
    for i in range(future_data.shape[0]):
        hist_df = pd.concat([train_data, future_data.iloc[:i]], axis=0)

        model = VAR(hist_df)
        fitted = model.fit(lag_order)

        # One-step forecast
        forecast_i = fitted.forecast(hist_df.values[-lag_order:], steps=1)[0]
        pred_series = pd.Series(forecast_i, index=hist_df.columns)
        pred_series['ds'] = future_data.iloc[i]['ds']
        preds.append(pred_series)

    return pd.DataFrame(preds)


###############################################
# 🔹 VALIDATION FORECAST
###############################################
val_preds = rolling_var_forecast(train, val, lag_order)

val_merge = val.reset_index().merge(val_preds[['ds', 'y']], on='ds', how='left')
val_merge = val_merge.set_index('ds')

val_mae  = np.mean(np.abs(val_merge['y_x'] - val_merge['y_y']))
val_mape = np.mean(np.abs((val_merge['y_x'] - val_merge['y_y']) / val_merge['y_x']))

print("\n=== VAR VALIDATION PERFORMANCE ===")
print("MAE :", round(val_mae, 3))
print("MAPE:", round(val_mape, 4))


###############################################
# 🔹 TEST FORECAST
###############################################
full_hist = pd.concat([train, val])
test_preds = rolling_var_forecast(full_hist, test, lag_order)

test_merge = test.reset_index().merge(test_preds[['ds','y']], on='ds', how='left')
test_merge = test_merge.set_index('ds')

test_mae  = np.mean(np.abs(test_merge['y_x'] - test_merge['y_y']))
test_mape = np.mean(np.abs((test_merge['y_x'] - test_merge['y_y']) / test_merge['y_x']))

print("\n=== VAR TEST PERFORMANCE ===")
print("MAE :", round(test_mae, 3))
print("MAPE:", round(test_mape, 4))
