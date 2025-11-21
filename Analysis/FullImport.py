

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

FinalGeorgiaCombinedData.to_csv("FinalFloridaCombinedData.csv", index=True)