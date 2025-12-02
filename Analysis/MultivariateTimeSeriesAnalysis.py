

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



FinalFloridaCombinedData = pd.read_csv("Final_GA_CombinedData.csv")

FinalFloridaCombinedData.head()

FinalFloridaCombinedData = FinalFloridaCombinedData.drop(columns=["Active"])

FinalFloridaCombinedData = FinalFloridaCombinedData.set_index('Date', append=True) 


varModelSelect = VAR(FinalFloridaCombinedData).select_order(maxlags=10)
print(varModelSelect.summary())

varModel = VAR(FinalFloridaCombinedData).fit(maxlags=4)
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
coef_Claims = step(FinalFloridaCombinedData, steps=5, maxLag=2, colName='claims') 
print(f"Claims Model {coef_Claims}")
coef_Covid = step(FinalFloridaCombinedData, steps=5, maxLag=2, colName='Confirmed') 
print(f"Covid Model {coef_Covid}")
