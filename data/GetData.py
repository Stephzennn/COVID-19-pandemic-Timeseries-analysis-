# COVID-19 Data (daily reports)




import pandas as pd
#from pandas_datareader import data as pdr
#import datetime as dt
#from datetime import datetime
#from datetime import timedelta
#import numpy as np
from GetCovidData import getData
"""
#Load COVID-19 dataset
covid_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/04-30-2020.csv"

covid_url2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/03-11-2021.csv"

covid_df = pd.read_csv(covid_url)

covid_df2 = pd.read_csv(covid_url2)


dd = np.hstack((covid_df['Province_State'].values.reshape(-1, 1),
                      covid_df['Confirmed'].values.reshape(-1, 1),
                      covid_df2['Confirmed'].values.reshape(-1, 1)))

dd1 = pd.DataFrame(dd, columns=["Province", "03-20-2021", "03-11-2021"])
dd1

"""

"""
['Province_State', 'Country_Region', 'Last_Update', 'Lat', 'Long_',
       'Confirmed', 'Deaths', 'Recovered', 'Active', 'FIPS', 'Incident_Rate',
       'Total_Test_Results', 'People_Hospitalized', 'Case_Fatality_Ratio',
       'UID', 'ISO3', 'Testing_Rate', 'Hospitalization_Rate', 'Date',
       'People_Tested', 'Mortality_Rate']
"""



df = getData(column='Deaths')

df.columns = ['Province_State'] + list(pd.to_datetime(df.columns[1:], format="%m-%d-%Y"))

df = df.set_index('Province_State')

print(df.head())

newYork = df.loc['New York'] 

'2023-03-18'

df_sum = (
    df.drop(columns=['Province_State'], errors='ignore')
      .sum()
      .reset_index()
      .rename(columns={'index': 'Date', 0: 'Total_Confirmed'})
)

df_mean = (
    df.drop(columns=['Province_State', 'Province'], errors='ignore')
      .select_dtypes(include='number')                  
      .mean()                                           
      .reset_index()
      .rename(columns={'index': 'Date', 0: 'Average_Confirmed'})
)

print(df_mean.head())

import requests
import pandas as pd
import certifi

url = "https://data.cdc.gov/resource/pwn4-m3yp.csv"

response = requests.get(url, verify=certifi.where())
response.raise_for_status()

from io import StringIO
covid_cdc = pd.read_csv(StringIO(response.text))

print("Loaded:", covid_cdc.shape)
print(covid_cdc.head())



first_col = covid_cdc.columns[0]
print("Aggregating by:", first_col)

byState = covid_cdc.groupby(first_col).sum(numeric_only=True)

byState.columns
byState.index = pd.to_datetime(byState.index)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(
    byState.index,
    byState['new_deaths'],
    color='orange',
    linewidth=2,
    marker='o',
    label='COVID-19: Aggregated USA Weekly New Deaths'
)

plt.title("COVID-19: Aggregated USA Weekly New Deaths", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Deaths", fontsize=12)
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

plt.show()



plt.figure(figsize=(12, 6))
plt.plot(
    byState.index,
    byState['new_cases'],
    color='orange',
    linewidth=2,
    marker='o',
    label='COVID-19: Aggregated USA Weekly New Cases'
)

plt.title("COVID-19: Aggregated USA Weekly New Cases", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Cases", fontsize=12)
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

plt.show()