import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import yaml

# ============================================
# CONFIGURATION & Dynamic Splitting Function
# ============================================

CONFIG = {
    'input_file': 'FL_GA_NY.csv',
    'test_weeks': 16,  # Last 16 weeks for testing
    'date_column': 'Date',
    'state_column': 'Province_State',
    'confirmed_column': 'Confirmed',
    'deaths_column': 'Deaths',
    'claims_column': 'claims', 
    'recovered_column': 'Recovered', 
    'active_column': 'Active',  
    'output_wide': 'multi_state_panel_wide.csv',
    'output_train_wide': 'train_data.csv',
    'output_test_wide': 'test_data.csv',
    'output_train_long': 'train_long.csv',
    'output_test_long': 'test_long.csv',
    'output_viz': 'train_test_split_visualization.png',
    'output_train_nf': 'train_neuralforecast.csv',
    'output_test_nf': 'test_neuralforecast.csv'

}

# ============================================
# LOAD DATA & DYNAMIC MULTI-STATE DATA PROCESSING
# ============================================

print("DYNAMIC MULTI-STATE DATA PROCESSING")

print(f"\nLoading data from: {CONFIG['input_file']}")
df = pd.read_csv(CONFIG['input_file'])

print(f"\nOriginal data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\nFirst few rows:")
print(df.head(5))

# ============================================
# AUTO-DETECT STATES
# ============================================

print(f"\n{'='*80}")
print("AUTO-DETECTING STATES")
print(f"{'='*80}")

# Get unique states from Province_State column
states = sorted(df[CONFIG['state_column']].unique())

print(f"\n AUTO-DETECTED STATES: {states}")
print(f"   Found {len(states)} states in dataset")

# Count observations per state
print("\nObservations per state:")
for state in states:
    count = len(df[df[CONFIG['state_column']] == state])
    print(f"   {state}: {count} weeks")

# ============================================
# CHECK FOR DUPLICATES
# ============================================

print(f"\n{'='*80}")
print("CHECKING FOR DUPLICATES")
print(f"{'='*80}")

# Check if there are duplicate date-state combinations
duplicates = df.groupby([CONFIG['date_column'], CONFIG['state_column']]).size()
max_duplicates = duplicates.max()

if max_duplicates > 1:
    print(f"\n WARNING: Found {max_duplicates} duplicate rows per date-state combination!")
    print("Removing duplicates (keeping first occurrence)...")
    
    df = df.drop_duplicates(subset=[CONFIG['date_column'], CONFIG['state_column']], keep='first')
    print(f" Removed duplicates. New shape: {df.shape}")
else:
    print(" No duplicates found")

# ============================================
# PROCESS DATES
# ============================================

print(f"\n{'='*80}")
print("PROCESSING DATES")
print(f"{'='*80}")

df[CONFIG['date_column']] = pd.to_datetime(df[CONFIG['date_column']])
df = df.sort_values([CONFIG['state_column'], CONFIG['date_column']]).reset_index(drop=True)

print(f"Date range: {df[CONFIG['date_column']].min()} to {df[CONFIG['date_column']].max()}")

# Get unique dates
unique_dates = sorted(df[CONFIG['date_column']].unique())
print(f"Unique dates: {len(unique_dates)}")

# Verify each state has same dates
print("\nVerifying date consistency across states:")
for state in states:
    state_dates = len(df[df[CONFIG['state_column']] == state])
    print(f"  {state}: {state_dates} observations")
    
    if state_dates != len(unique_dates):
        print(f"WARNING: {state} has different number of dates!")

# ============================================
# CONVERT CUMULATIVE TO WEEKLY NEW
# ============================================

print(f"\n{'='*80}")
print("CONVERTING CUMULATIVE TO WEEKLY NEW CASES/DEATHS")
print(f"{'='*80}")

# Process each state separately (important for diff())
df_processed_list = []

for state in states:
    print(f"\nProcessing {state}...")
    
    # Get data for this state
    state_df = df[df[CONFIG['state_column']] == state].copy()
    state_df = state_df.sort_values(CONFIG['date_column']).reset_index(drop=True)
    
    # Convert cumulative to weekly NEW
    if CONFIG['confirmed_column'] in state_df.columns:
        state_df['new_cases'] = state_df[CONFIG['confirmed_column']].diff()
        state_df.loc[0, 'new_cases'] = state_df.loc[0, CONFIG['confirmed_column']]  # First week
        state_df['new_cases'] = state_df['new_cases'].clip(lower=0)
        print(f"  Created new_cases from {CONFIG['confirmed_column']}")
    
    if CONFIG['deaths_column'] in state_df.columns:
        state_df['new_deaths'] = state_df[CONFIG['deaths_column']].diff()
        state_df.loc[0, 'new_deaths'] = state_df.loc[0, CONFIG['deaths_column']]  # First week
        state_df['new_deaths'] = state_df['new_deaths'].clip(lower=0)
        print(f"  Created new_deaths from {CONFIG['deaths_column']}")
    
    # Check for claims column
    if CONFIG['claims_column'] not in state_df.columns:
        print(f"WARNING: {CONFIG['claims_column']} not found!")
    
    df_processed_list.append(state_df)

# Combine all states back together
df_long = pd.concat(df_processed_list, ignore_index=True)
df_long = df_long.sort_values([CONFIG['state_column'], CONFIG['date_column']]).reset_index(drop=True)

print(f"\n Processed all states")
print(f"Total observations: {len(df_long)} ({len(df_long)//len(states)} weeks × {len(states)} states)")

# ============================================
# CREATE CLEAN LONG FORMAT
# ============================================

print(f"\n{'='*80}")
print("CREATING CLEAN LONG FORMAT")
print(f"{'='*80}")

# Select and rename columns
df_long_clean = pd.DataFrame({
    'date': df_long[CONFIG['date_column']],
    'state': df_long[CONFIG['state_column']],
    'claims': df_long[CONFIG['claims_column']],
    'cases': df_long['new_cases'],
    'deaths': df_long['new_deaths']
})

print(f"\nClean long format shape: {df_long_clean.shape}")
print(f"Expected: {len(unique_dates) * len(states)} rows")

print("\nFirst 9 rows (3 weeks × 3 states):")
print(df_long_clean.head(6))

# Verify no duplicates in clean data
dup_check = df_long_clean.groupby(['date', 'state']).size()
if dup_check.max() > 1:
    print("\n ERROR: Still have duplicates in clean data!")
    print("Removing duplicates...")
    df_long_clean = df_long_clean.drop_duplicates(subset=['date', 'state'], keep='first')
else:
    print("\n No duplicates in clean long format")

# ============================================
# PIVOT TO WIDE FORMAT
# ============================================

print(f"\n{'='*80}")
print("CONVERTING TO WIDE FORMAT")
print(f"{'='*80}")

# Pivot each metric separately
print("\nPivoting claims...")
claims_wide = df_long_clean.pivot(index='date', columns='state', values='claims')
claims_wide.columns = [f'{col}_claims' for col in claims_wide.columns]

print("Pivoting cases...")
cases_wide = df_long_clean.pivot(index='date', columns='state', values='cases')
cases_wide.columns = [f'{col}_cases' for col in cases_wide.columns]

print("Pivoting deaths...")
deaths_wide = df_long_clean.pivot(index='date', columns='state', values='deaths')
deaths_wide.columns = [f'{col}_deaths' for col in deaths_wide.columns]

# Combine all pivoted dataframes
df_wide = pd.concat([claims_wide, cases_wide, deaths_wide], axis=1)
df_wide = df_wide.reset_index()

# Reorder columns for readability (date, then all FL, then all GA, then all NY)
ordered_cols = ['date']
for state in sorted(states):
    ordered_cols.extend([f'{state}_claims', f'{state}_cases', f'{state}_deaths'])

df_wide = df_wide[ordered_cols]

print(f"\nWide format shape: {df_wide.shape}")
print(f"Expected: {len(unique_dates)} rows × {1 + len(states)*3} columns")

print("\nFirst 5 rows:")
print(df_wide.head())

# Verify no duplicate dates
if df_wide['date'].duplicated().any():
    print("\n⚠ ERROR: Duplicate dates in wide format!")
    print("Number of duplicates:", df_wide['date'].duplicated().sum())
    print("\nRemoving duplicate dates (keeping first)...")
    df_wide = df_wide.drop_duplicates(subset=['date'], keep='first')
    print(f"✓ New shape: {df_wide.shape}")
else:
    print("\n✓ No duplicate dates in wide format")

# ============================================
# DATA QUALITY CHECKS
# ============================================

print(f"\n{'='*80}")
print("DATA QUALITY CHECKS")
print(f"{'='*80}")

# Check for missing values
print("\nMissing values in wide format:")
missing = df_wide.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values")

# Check for negative values
print("\nChecking for negative values:")
has_negatives = False
for col in df_wide.columns:
    if col != 'date':
        neg_count = (df_wide[col] < 0).sum()
        if neg_count > 0:
            print(f" {col}: {neg_count} negative values")
            has_negatives = True

if not has_negatives:
    print("  No negative values detected")    

# ============================================
# TRAIN/TEST SPLIT
# ============================================

print(f"\n{'='*80}")
print(f"TRAIN/TEST SPLIT - Last {CONFIG['test_weeks']} Weeks as Test")
print(f"{'='*80}")

total_weeks = len(df_wide)
test_size = CONFIG['test_weeks']
train_size = total_weeks - test_size

print(f"\nTotal weeks: {total_weeks}")
print(f"Train size: {train_size} weeks ({train_size/total_weeks*100:.1f}%)")
print(f"Test size: {test_size} weeks ({test_size/total_weeks*100:.1f}%)")

# Split wide format
train_data_wide = df_wide.iloc[:train_size].copy()
test_data_wide = df_wide.iloc[train_size:].copy()

# Split long format
train_dates = train_data_wide['date'].tolist()
test_dates = test_data_wide['date'].tolist()

train_data_long = df_long_clean[df_long_clean['date'].isin(train_dates)].copy()
test_data_long = df_long_clean[df_long_clean['date'].isin(test_dates)].copy()

# Verify splits
print(f"\nWIDE FORMAT:")
print(f"  Train: {len(train_data_wide)} weeks")
print(f"  Test: {len(test_data_wide)} weeks")

print(f"\nLONG FORMAT:")
print(f"  Train: {len(train_data_long)} rows ({len(train_data_long)//len(states)} weeks × {len(states)} states)")
print(f"  Test: {len(test_data_long)} rows ({len(test_data_long)//len(states)} weeks × {len(states)} states)")

print(f"\nDate ranges:")
print(f"  Train: {train_data_wide['date'].min()} to {train_data_wide['date'].max()}")
print(f"  Test: {test_data_wide['date'].min()} to {test_data_wide['date'].max()}")

# ============================================
# SAVE PROCESSED DATA
# ============================================

print(f"\n{'='*80}")
print("SAVING PROCESSED DATA")
print(f"{'='*80}")

# Wide format
df_wide.to_csv(CONFIG['output_wide'], index=False)
print(f" {CONFIG['output_wide']}")

train_data_wide.to_csv(CONFIG['output_train_wide'], index=False)
print(f" {CONFIG['output_train_wide']}")

test_data_wide.to_csv(CONFIG['output_test_wide'], index=False)
print(f" {CONFIG['output_test_wide']}")

# Long format
train_data_long.to_csv(CONFIG['output_train_long'], index=False)
print(f" {CONFIG['output_train_long']}")

test_data_long.to_csv(CONFIG['output_test_long'], index=False)
print(f" {CONFIG['output_test_long']}")

# NeuralForecast format
nf_train = train_data_long.rename(columns={'state': 'unique_id', 'date': 'ds', 'claims': 'y'})
nf_train = nf_train[['unique_id', 'ds', 'y', 'cases', 'deaths']]

nf_test = test_data_long.rename(columns={'state': 'unique_id', 'date': 'ds', 'claims': 'y'})
nf_test = nf_test[['unique_id', 'ds', 'y', 'cases', 'deaths']]

nf_train.to_csv(CONFIG['output_train_nf'], index=False)
nf_test.to_csv(CONFIG['output_test_nf'], index=False)
print(f" {CONFIG['output_train_nf']}")
print(f" {CONFIG['output_test_nf']}")

# States list
with open('states_list.txt', 'w') as f:
    f.write(','.join(states))
print(f" states_list.txt")

# ============================================
# VISUALIZATION
# ============================================

print(f"\n{'='*80}")
print("CREATING VISUALIZATION")
print(f"{'='*80}")

n_states = len(states)
fig, axes = plt.subplots(n_states, 2, figsize=(16, 5*n_states))

if n_states == 1:
    axes = axes.reshape(1, -1)

for idx, state in enumerate(states):
    # Filter data for this state
    state_full = df_long_clean[df_long_clean['state'] == state]
    state_train = train_data_long[train_data_long['state'] == state]
    state_test = test_data_long[test_data_long['state'] == state]
    
    # Plot claims
    ax = axes[idx, 0]
    ax.plot(state_full['date'], state_full['claims'], 
            color='lightgray', alpha=0.5, linewidth=1)
    ax.plot(state_train['date'], state_train['claims'],
            color='blue', linewidth=2.5, label=f'Train ({len(state_train)} weeks)', 
            marker='o', markersize=3)
    ax.plot(state_test['date'], state_test['claims'],
            color='red', linewidth=2.5, label=f'Test ({len(state_test)} weeks)', 
            marker='s', markersize=4)
    
    split_date = train_data_wide['date'].max()
    ax.axvline(split_date, color='black', linestyle='--', linewidth=2)

    ax.set_title(f'{state} - Unemployment Claims', fontsize=14, fontweight='bold')
    ax.set_ylabel('Claims (thousands)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Plot cases
    ax2 = axes[idx, 1]
    ax2.plot(state_full['date'], state_full['cases'], 
             color='lightgray', alpha=0.5, linewidth=1)
    ax2.plot(state_train['date'], state_train['cases'],
             color='blue', linewidth=2.5, label=f'Train ({len(state_train)} weeks)', 
             marker='o', markersize=3)
    ax2.plot(state_test['date'], state_test['cases'],
             color='red', linewidth=2.5, label=f'Test ({len(state_test)} weeks)', 
             marker='s', markersize=4)
    
    ax2.axvline(split_date, color='black', linestyle='--', linewidth=2)
    
    ax2.set_title(f'{state} - COVID Cases', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Weekly New Cases')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(CONFIG['output_viz'], dpi=300, bbox_inches='tight')
print(f"✓ Saved {CONFIG['output_viz']}")
plt.show()

# ============================================
# SUMMARY
# ============================================

print(f"\n{'='*80}")
print("DATA PREPARATION COMPLETE")
print(f"{'='*80}")

print(f"\n Successfully processed {len(states)} states: {', '.join(states)}")
print(f" Wide format: {len(df_wide)} weeks (NO DUPLICATES)")
print(f" Long format: {len(df_long_clean)} observations")
print(f" Train: {train_size} weeks, Test: {test_size} weeks")

