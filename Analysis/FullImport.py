

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

    # Save to CSV
    output_name = f"Final_{state}_CombinedData.csv"
    final_df.to_csv(output_name, index=True)

    print(f"Saved combined data → {output_name}")
    return final_df



def build_multiple_states(states, extra_columns=['Deaths', 'Recovered', 'Active']):
    """ Accepts a single state string or a list of state abbreviations. """
    
    if isinstance(states, str):
        return build_state_combined_data(states, extra_columns)

    results = {}
    for st in states:
        print(f"\nProcessing: {st}")
        results[st] = build_state_combined_data(st, extra_columns)

    print("\nDone with all states!")
    return results


#Usage

if __name__ == "__main__":
    # Single state call
    df_FL = build_multiple_states('GA')

    # Multi-state call
    df_dict = build_multiple_states(['FL', 'GA', 'NY'])
