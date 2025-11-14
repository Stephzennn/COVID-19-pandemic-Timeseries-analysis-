from fredapi import Fred
import pandas as pd

fred = Fred(api_key='API_KEY_HERE')

#fred = Fred(api_key='f9a44139decd5e780297cade865dd2eb')
# TL's key: f9a44139decd5e780297cade865dd2eb

from FredAPIKey import fred

state_mapping = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands'
}

states = list(state_mapping.keys())

# Download initial state-level claims (ICSA = Initial Claims by State)
print('Downloading Initial Claims data...')
state_claims = {}
for state in states:
    series_id = f'{state}ICLAIMS'
    try:
        state_claims[state] = fred.get_series(series_id,
                                                observation_start='2020-02-28',
                                                observation_end='2023-03-18')
        print(f'Successfully retrieved data for {state}')
    except:
        print(f'Failed to retrieve data for {state}')

# Combine all state data into a single DataFrame
df_state_claims = pd.DataFrame(state_claims)
df_state_claims.to_csv('state_unemployment_claims.csv')

# Melt the DataFrame to long format
df_melted = df_state_claims.reset_index().melt(id_vars='index', var_name='Province_State', value_name='Initial_Claims')

df_melted.rename(columns={'index': 'Date'}, inplace=True)
df_melted.to_csv('state_unemployment_claims_long.csv', index=False)

