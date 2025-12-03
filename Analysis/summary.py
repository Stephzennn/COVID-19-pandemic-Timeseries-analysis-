
import os
os.chdir("..")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


FinalFloridaCombinedData = pd.read_csv("FinalFloridaCombinedData.csv")

FinalGeorgiaCombinedData = pd.read_csv("Final_GA_CombinedData.csv")


cols_to_keep = ["Date", "claims", "Confirmed", "Deaths"]

FinalFloridaCombinedData = FinalFloridaCombinedData[cols_to_keep]
FinalGeorgiaCombinedData = FinalGeorgiaCombinedData[cols_to_keep]


FinalFloridaCombinedData['Date'] = pd.to_datetime(FinalFloridaCombinedData['Date'])
FinalGeorgiaCombinedData['Date'] = pd.to_datetime(FinalGeorgiaCombinedData['Date'])



def create_summary_table(df, state_name):
   
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print(df.columns)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    print(df.columns)
    summary = numeric_df.describe().T  
    summary = summary[['count', 'mean', 'std', 'min', 'max']] 
    
    summary = summary.reset_index()
    summary.rename(columns={'index': 'Variable'}, inplace=True)
    
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')
    
    summary['State'] = state_name
    summary['Time Period'] = f"{start_date} to {end_date}"
    
    return summary

# Generate summaries
fl_summary = create_summary_table(FinalFloridaCombinedData, 'Florida')
ga_summary = create_summary_table(FinalGeorgiaCombinedData, 'Georgia')

# Display 
fl_summary


# to CSV files
fl_summary.to_csv("Florida_Summary_Table.csv", index=False)
ga_summary.to_csv("Georgia_Summary_Table.csv", index=False)