

import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime
from datetime import timedelta
import numpy as np

import os



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
