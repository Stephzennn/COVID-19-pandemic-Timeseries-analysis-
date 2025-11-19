

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
from statsmodels.graphics.tsaplots import plot_accf_grid
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.gam.api import GLMGam
from scipy import stats
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=RuntimeWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning, module='statsmodels')

from data.GetCombinedData import getCombinedData





GeorgiaCombinedData = getCombinedData(state='GA')

GeorgiaCombinedData  = GeorgiaCombinedData.set_index('claims', append=True) 
#'Deaths', 'Recovered', 'Active'
GeorgiaCombinedData.columns
GeorgiaCombinedDataDeath = getCombinedData(state='GA',column='Deaths')

GeorgiaCombinedDataDeath = GeorgiaCombinedDataDeath.set_index('claims', append=True) 


GeorgiaCombinedDataRecovered = getCombinedData(state='GA', column='Recovered')


GeorgiaCombinedDataRecovered = GeorgiaCombinedDataRecovered.set_index('claims', append=True) 


GeorgiaCombinedDataActive = getCombinedData(state='GA', column='Active')

GeorgiaCombinedDataActive = GeorgiaCombinedDataActive.set_index('claims', append=True) 



FinalGeorgiaCombinedData = GeorgiaCombinedData.join([GeorgiaCombinedDataDeath], how='inner')

FinalGeorgiaCombinedData.head()
"""
FinalGeorgiaCombinedData = pd.concat(
    [GeorgiaCombinedData, GeorgiaCombinedDataDeath, GeorgiaCombinedDataRecovered, GeorgiaCombinedDataActive],
    axis=1,
    join="inner"
)
"""

FinalGeorgiaCombinedData = FinalGeorgiaCombinedData.reset_index()

FinalGeorgiaCombinedData.columns = ['Date', 'claims', 'Confirmed', 'Deaths']

FinalGeorgiaCombinedData = FinalGeorgiaCombinedData.set_index('Date', append=True) 

FinalGeorgiaCombinedData.head()

varModelSelect = VAR(FinalGeorgiaCombinedData).select_order(maxlags=10)
print(varModelSelect.summary())

varModel = VAR(FinalGeorgiaCombinedData).fit(maxlags=4)
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

coef_Claims = step(GeorgiaCombinedData, steps=5, maxLag=2, colName='claims') 
print(f"Claims Model {coef_Claims}")
coef_Covid = step(GeorgiaCombinedData, steps=5, maxLag=2, colName='covid') 
print(f"Covid Model {coef_Covid}")
