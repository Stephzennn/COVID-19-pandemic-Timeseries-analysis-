from GetCovidData import getData

from FredAPIKey import fred

import datetime as dt

import pandas as pd

import us 
"""
state = 'NY'
series_id = f'{state}ICLAIMS'

NY_claim= fred.get_series(series_id,
                                                observation_start='2020-02-28',
                                                observation_end='2023-03-18')

"""


def fix_date(date_str):
    try:
        # parse MM-DD-YYYY
        d = dt.datetime.strptime(date_str, "%m-%d-%Y")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected MM-DD-YYYY.")
    
    # Return in FRED format
    return d.strftime("%Y-%m-%d")

def getCombinedData(state = 'NY', column='Confirmed', startDate="04-12-2020", endDate="03-09-2023"):
    full_name = us.states.lookup(state)
    if full_name is None:
        raise ValueError(f"Invalid state abbreviation: {state}")
    full_name = full_name.name
    series_id = f'{state}ICLAIMS'
    observation_start = fix_date(startDate)
    observation_end   = fix_date(endDate)
    claim= fred.get_series(series_id,
                                                observation_start=observation_start,
                                                observation_end=observation_end)
    df = getData(column, startDate, endDate )
    df.columns = ['Province_State'] + list(pd.to_datetime(df.columns[1:], format="%m-%d-%Y"))
    df = df.set_index('Province_State')
    congruentDF = df.loc[full_name] 
    return claim, congruentDF
    

claim, congruentDF = getCombinedData()

print("wdwdqw")