import os
parent = os.path.dirname(os.getcwd())
data_path = os.path.join(parent, "data")

import sys

# Add to path if not already present
if data_path not in sys.path:

    sys.path.append(data_path)

from data.GetCovidData import getData
from data.FredAPIKey import fred
import datetime as dt
import pandas as pd
import us 




def fix_date(date_str):
    try:
        # parse MM-DD-YYYY
        d = dt.datetime.strptime(date_str, "%m-%d-%Y")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected MM-DD-YYYY.")
    
    # Return in FRED format
    return d.strftime("%Y-%m-%d")

# Comments on the functions were provided by Chat GPT
    
def getCombinedData(state='NY', column='Confirmed', startDate="04-12-2020", endDate="03-09-2023"):
    # Convert a state abbreviation to the full state name (for example, "NY" to "New York")
    full_name = us.states.lookup(state)
    if full_name is None:
        raise ValueError(f"Invalid state abbreviation: {state}")
    full_name = full_name.name

    # Build the FRED series ID for unemployment claims (for example, "NYICLAIMS")
    series_id = f'{state}ICLAIMS'

    # Convert dates from MM-DD-YYYY to the required FRED format YYYY-MM-DD
    observation_start = fix_date(startDate)
    observation_end   = fix_date(endDate)

    # Retrieve the unemployment claims time series from FRED
    claim = fred.get_series(series_id,
                            observation_start=observation_start,
                            observation_end=observation_end)

    # Retrieve the COVID time series for all states
    df = getData(column, startDate, endDate)

    # Convert all date columns to datetime objects and set Province_State as the index
    df.columns = ['Province_State'] + list(pd.to_datetime(df.columns[1:], format="%m-%d-%Y"))
    df = df.set_index('Province_State')

    # Extract the COVID time series for the selected state
    congruentDF = df.loc[full_name]

    # Combine the unemployment claims and COVID series using only the dates that appear in both
    joined = pd.concat([claim, congruentDF], axis=1, join='inner')
    joined.columns = ['claims', 'covid']  # Rename columns for clarity

    return joined

GeorgiaCombinedData = getCombinedData(state='GA')


