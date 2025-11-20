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

# Save to CSV

# If the csv exists, skip the following line

#FinalGeorgiaCombinedData.to_csv("FinalFloridaCombinedData.csv", index=True)
