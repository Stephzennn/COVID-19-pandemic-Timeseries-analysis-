# COVID-19 Data (daily reports)




import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime
from datetime import timedelta
import numpy as np

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


def getData(column='Confirmed', startDate="04-12-2020", endDate="03-09-2023"):
    try:
        minDate = datetime.strptime("04-12-2020", "%m-%d-%Y")
        maxDate = datetime.strptime("03-09-2023", "%m-%d-%Y")
        startDateTime = max(datetime.strptime(startDate, "%m-%d-%Y"), minDate)
        endDateTime = min(datetime.strptime(endDate, "%m-%d-%Y"), maxDate)
    except Exception as e:
        print("Error parsing dates:", e)
        return None

    all_data = []
    all_dates = []
    provinces_union = set()

    #Load all CSVs and collect data
    for offset in range((endDateTime - startDateTime).days + 1):
        day = startDateTime + timedelta(days=offset)
        url = (
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/csse_covid_19_daily_reports_us/"
            + day.strftime("%m-%d-%Y") + ".csv"
        )
        try:
            df = pd.read_csv(url)
            if column not in df.columns:
                print(f"{day.strftime('%m-%d-%Y')} missing '{column}' column.")
                continue
            df = df[['Province_State', column]].rename(columns={column: day.strftime("%m-%d-%Y")})
            all_data.append(df)
            all_dates.append(day.strftime("%m-%d-%Y"))
            provinces_union.update(df['Province_State'])
        except Exception as e:
            print(f"Skipping {day.strftime('%m-%d-%Y')} — {type(e).__name__}: {e}")
            continue

    if not all_data:
        print("No data loaded.")
        return None

    #Merge all dataframes by Province_State
    full_df = pd.DataFrame({'Province_State': sorted(provinces_union)})
    for df in all_data:
        full_df = full_df.merge(df, on='Province_State', how='left')

    #Convert columns to numeric
    for col in full_df.columns[1:]:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    print(f"Successfully loaded {len(all_data)} file(s) covering {all_dates[0]} → {all_dates[-1]}")
    return full_df

df = getData(column='Deaths')

print(df.head())

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
import matplotlib.pyplot as plt

# Convert Date to datetime (for proper x-axis scaling)
df_sum['Date'] = pd.to_datetime(df_sum['Date'], format='%m-%d-%Y', errors='coerce')

# Plot
df_sum.plot(x='Date', y='Total_Confirmed', figsize=(10, 5), marker='o', color='tab:blue')

plt.title('COVID-19 Total Confirmed Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Confirmed Cases')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Ensure the Date column is in datetime format
df_mean['Date'] = pd.to_datetime(df_mean['Date'], format='%m-%d-%Y', errors='coerce')

# Drop any rows where conversion failed (just in case)
df_mean = df_mean.dropna(subset=['Date'])

# Sort by date
df_mean = df_mean.sort_values('Date')

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(df_mean['Date'], df_mean['Average_Confirmed'], color='darkorange', marker='o', linewidth=2, label='Average Confirmed Cases')

# Add titles and labels
plt.title('Average COVID-19 Confirmed Cases per State Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Confirmed Cases', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

df_mean['Daily_Change'] = df_mean['Average_Confirmed'].diff()


plt.figure(figsize=(10, 5))
plt.plot(
    df_mean['Date'], 
    df_mean['Daily_Change'], 
    color='orange', 
    linewidth=2, 
    marker='o', 
    label='Daily Change'
)

plt.title('Day-to-Day Change in Average COVID-19 Deaths', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Change in Deaths', fontsize=12)
plt.axhline(0, color='gray', linestyle='--', linewidth=1) 
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Extract the series
series = df_mean['Average_Confirmed']

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(series, ax=axes[0], lags=30, title='Autocorrelation (ACF)')
plot_pacf(series, ax=axes[1], lags=30, title='Partial Autocorrelation (PACF)', method='ywm')

plt.tight_layout()
plt.show()