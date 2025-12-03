

import os

os.chdir(".")

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data.GetCombinedData import getCombinedData

from data.GetCombinedData import getCombinedData
import pandas as pd


def build_state_combined_data(state, extra_columns=['Deaths', 'Recovered', 'Active']):
   

    # Base data 
    final_df = getCombinedData(state=state).set_index('claims', append=True)

    # Fetch each extra metric one by one and join
    for col in extra_columns:
        try:
            temp_df = getCombinedData(state=state, column=col).set_index('claims', append=True)
            final_df = final_df.join(temp_df, how='inner')
        except Exception as e:
            print(f"Warning: {col} missing for {state} — SKIPPED ({type(e).__name__})")

    # Reset index and rename first two columns
    final_df = final_df.reset_index()
    final_df.rename(columns={final_df.columns[0]: 'Date', final_df.columns[1]: 'claims'}, inplace=True)

   
    final_df.set_index('Date', inplace=True)
    # Add state name
    final_df['Province_State'] = state

    # Weekly new cases & deaths
    final_df['New_Weekly_Cases'] = final_df['Confirmed'].diff().fillna(0)
    final_df['New_Weekly_Deaths'] = final_df['Deaths'].diff().fillna(0)

    # Reorder columns (optional)
    cols = ['Date', 'Province_State', 'claims', 'Confirmed', 'Deaths', 
        'New_Weekly_Cases', 'New_Weekly_Deaths', 'Recovered', 'Active']
    # Only keep columns that exist (Recovered/Active may be missing)
    final_df = final_df[[col for col in cols if col in final_df.columns]]

    # Save to CSV
    output_name = f"Final_{state}_CombinedData.csv"
    final_df.to_csv(output_name, index=True)

    print(f"Saved combined data → {output_name}")
    return final_df



def build_multiple_states(states, extra_columns=['Deaths', 'Recovered', 'Active']):
    """ Accepts a single state string or a list of state abbreviations. """
    
    # Normalize input to always be a list
    if isinstance(states, str):
        states = [states]

    results = {}

    for st in states:
        print(f"\nProcessing: {st}")
        results[st] = build_state_combined_data(st, extra_columns)

    print("\n✔ Done with all states!")
    return results

#Usage

if __name__ == "__main__":
    # Choose the states you want
    states_to_build = ['FL','GA','NY','CA','TX','WA','IL']

    # Build multiple states at once
    results = build_multiple_states(states_to_build)

    print("All states completed! Files saved:")
    for state in results:
        print(f"  Final_{state}_CombinedData.csv")
