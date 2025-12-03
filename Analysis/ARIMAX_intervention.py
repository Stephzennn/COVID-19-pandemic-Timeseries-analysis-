import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

"""
ARIMAX with Intervention Analysis
Purpose: Explicitly model pandemic policy interventions
Methods: Add intervention dummy variables for:
  - Initial shutdown
  - Reopening
  - Delta/Omicron waves
  - Vaccine rollout
"""

# Create output directory
os.makedirs('Results/intervention', exist_ok=True)

print("="*80)
print("ARIMAX WITH INTERVENTION ANALYSIS")
print("="*80)

# ============================================
# LOAD DATA
# ============================================

print("\nLoading data...")
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

with open('states_list.txt', 'r') as f:
    states = f.read().strip().split(',')

print(f"States: {states}")
print(f"Train: {len(train_data)} weeks ({train_data['date'].min()} to {train_data['date'].max()})")
print(f"Test: {len(test_data)} weeks ({test_data['date'].min()} to {test_data['date'].max()})")

# ============================================
# DEFINE INTERVENTION PERIODS
# ============================================

# Key pandemic intervention periods
interventions = {
    'initial_shutdown': ('2020-03-15', '2020-05-31'),
    'reopening': ('2020-06-01', '2020-08-31'),
    'second_wave': ('2020-11-01', '2021-02-28'),
    'vaccine_rollout': ('2021-01-01', '2021-06-30'),
    'delta_wave': ('2021-07-01', '2021-10-31'),
    'omicron_wave': ('2021-12-01', '2022-02-28')
}

print(f"\n{'='*80}")
print("DEFINED INTERVENTION PERIODS")
print(f"{'='*80}")
for name, (start, end) in interventions.items():
    print(f"  {name:20s}: {start} to {end}")

# ============================================
# CREATE INTERVENTION DUMMIES
# ============================================

def create_intervention_dummies(dates, interventions):
    """Create dummy variables for each intervention period"""
    
    dummies = pd.DataFrame(index=dates.index)
    
    for name, (start, end) in interventions.items():
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        dummies[name] = ((dates >= start_date) & (dates <= end_date)).astype(int)
    
    return dummies

# Create intervention dummies for train and test
train_interventions = create_intervention_dummies(train_data['date'], interventions)
test_interventions = create_intervention_dummies(test_data['date'], interventions)

print(f"\nCreated {len(interventions)} intervention dummy variables")
print(f"\nTrain intervention counts (weeks in each period):")
print(train_interventions.sum())

# ============================================
# VISUALIZE INTERVENTIONS
# ============================================

print(f"\n{'='*80}")
print("CREATING INTERVENTION VISUALIZATION")
print(f"{'='*80}")

fig, axes = plt.subplots(len(states), 1, figsize=(16, 5*len(states)))

if len(states) == 1:
    axes = [axes]

for idx, state in enumerate(states):
    ax = axes[idx]
    
    # Plot claims
    ax.plot(train_data['date'], train_data[f'{state}_claims'],
            linewidth=2, label='Unemployment Claims', color='blue', zorder=5)
    
    # Shade intervention periods
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'purple']
    for (name, (start, end)), color in zip(interventions.items(), colors):
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        # Shade the intervention period
        ax.axvspan(start_date, end_date, alpha=0.3, color=color, label=name)
    
    ax.set_title(f'{state} - Unemployment Claims with Intervention Periods', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Unemployment Claims (thousands)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Results/intervention/intervention_periods_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/intervention/intervention_periods_visualization.png")
plt.close()

# ============================================
# FIT ARIMAX WITH INTERVENTIONS
# ============================================

print(f"\n{'='*80}")
print("FITTING ARIMAX MODELS WITH INTERVENTION VARIABLES")
print(f"{'='*80}")

results_list = []
coefficient_list = []

for state in states:
    print(f"\n{'='*60}")
    print(f"STATE: {state}")
    print(f"{'='*60}")
    
    # Get target variable
    y_train = train_data[f'{state}_claims']
    y_test = test_data[f'{state}_claims']
    
    # Exogenous variables: COVID features + intervention dummies
    X_train = pd.concat([
        train_data[[f'{state}_cases', f'{state}_deaths']],
        train_interventions
    ], axis=1)
    
    X_test = pd.concat([
        test_data[[f'{state}_cases', f'{state}_deaths']],
        test_interventions
    ], axis=1)
    
    print(f"\nTarget: {state}_claims")
    print(f"Exogenous variables ({X_train.shape[1]}):")
    print(f"  - {state}_cases")
    print(f"  - {state}_deaths")
    for intv in interventions.keys():
        print(f"  - {intv}")
    
    try:
        # Fit ARIMAX model
        print(f"\nFitting ARIMAX(2,1,0) with {X_train.shape[1]} exogenous variables...")
        
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(2, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False, maxiter=200)
        
        print(" Model fitted successfully")
        
        # Generate forecast
        forecast = fitted.forecast(steps=len(test_data), exog=X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        r2 = r2_score(y_test, forecast)
        
        # Calculate MAPE safely
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - forecast[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(y_test.values))
        pred_direction = np.sign(np.diff(forecast.values))
        direction_acc = np.mean(actual_direction == pred_direction) * 100
        
        # Store results
        results_list.append({
            'State': state,
            'Model': 'ARIMAX_Intervention',
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Acc': direction_acc,
            'AIC': fitted.aic,
            'BIC': fitted.bic,
            'Converged': fitted.mle_retvals['converged']
        })
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {state}")
        print(f"{'='*60}")
        print(f"  MAE:                {mae:,.0f}")
        print(f"  RMSE:               {rmse:,.0f}")
        print(f"  MAPE:               {mape:.2f}%")
        print(f"  R²:                 {r2:.3f}")
        print(f"  Direction Accuracy: {direction_acc:.1f}%")
        print(f"  AIC:                {fitted.aic:.0f}")
        print(f"  BIC:                {fitted.bic:.0f}")
        print(f"  Converged:          {fitted.mle_retvals['converged']}")
        
        # Extract intervention coefficients
        print(f"\n{'='*60}")
        print(f"INTERVENTION COEFFICIENTS FOR {state}")
        print(f"{'='*60}")
        
        params = fitted.params
        pvalues = fitted.pvalues
        
        for intv_name in interventions.keys():
            if intv_name in params.index:
                coef = params[intv_name]
                pval = pvalues[intv_name]
                significant = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                
                print(f"  {intv_name:20s}: {coef:10,.2f}  (p={pval:.4f}) {significant}")
                
                coefficient_list.append({
                    'State': state,
                    'Intervention': intv_name,
                    'Coefficient': coef,
                    'P_Value': pval,
                    'Significant': pval < 0.05
                })
        
        # Save individual forecast
        forecast_df = pd.DataFrame({
            'date': test_data['date'],
            'actual': y_test.values,
            'forecast': forecast.values,
            'error': y_test.values - forecast.values
        })
        forecast_df.to_csv(f'Results/intervention/forecast_{state}.csv', index=False)
        print(f"\n Saved forecast: Results/intervention/forecast_{state}.csv")
        
    except Exception as e:
        print(f"\n ERROR: Model failed for {state}")
        print(f"   {str(e)}")
        
        results_list.append({
            'State': state,
            'Model': 'ARIMAX_Intervention',
            'MAE': np.nan,
            'RMSE': np.nan,
            'R2': np.nan,
            'MAPE': np.nan,
            'Direction_Acc': np.nan,
            'AIC': np.nan,
            'BIC': np.nan,
            'Converged': False
        })

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

# Save main results
results_df = pd.DataFrame(results_list)
results_df.to_csv('Results/intervention/arimax_intervention_results.csv', index=False)
print(" Saved: Results/intervention/arimax_intervention_results.csv")

# Save intervention coefficients
if len(coefficient_list) > 0:
    intervention_coefficients = pd.DataFrame(coefficient_list)
    intervention_coefficients.to_csv('Results/intervention/intervention_coefficients.csv', index=False)
    print(" Saved: Results/intervention/intervention_coefficients.csv")
else:
    print(" No intervention coefficients to save")

# ============================================
# SUMMARY STATISTICS
# ============================================

print(f"\n{'='*80}")
print("SUMMARY - ARIMAX WITH INTERVENTIONS")
print(f"{'='*80}")

print("\nModel Performance:")
print(results_df[['State', 'MAE', 'RMSE', 'R2', 'Direction_Acc']].to_string(index=False))

if len(coefficient_list) > 0:
    print(f"\nIntervention Effects (Significant at p<0.05):")
    sig_interventions = intervention_coefficients[intervention_coefficients['Significant']]
    
    if len(sig_interventions) > 0:
        print(sig_interventions[['State', 'Intervention', 'Coefficient', 'P_Value']].to_string(index=False))
    else:
        print("  No interventions were statistically significant")

# ============================================
# VISUALIZATION: FORECASTS
# ============================================

print(f"\n{'='*80}")
print("CREATING FORECAST VISUALIZATIONS")
print(f"{'='*80}")

fig, axes = plt.subplots(len(states), 2, figsize=(16, 5*len(states)))

if len(states) == 1:
    axes = axes.reshape(1, -1)

for idx, state in enumerate(states):
    # Load forecast for this state
    try:
        forecast_data = pd.read_csv(f'Results/intervention/forecast_{state}.csv')
        forecast_data['date'] = pd.to_datetime(forecast_data['date'])
        
        # Left plot: Forecast vs Actual
        ax1 = axes[idx, 0]
        
        # Plot training data (context)
        ax1.plot(train_data['date'], train_data[f'{state}_claims'],
                 color='gray', alpha=0.3, linewidth=1, label='Training Data')
        
        # Plot test actual
        ax1.plot(forecast_data['date'], forecast_data['actual'],
                 color='black', linewidth=2.5, marker='o', markersize=5,
                 label='Actual', zorder=3)
        
        # Plot forecast
        ax1.plot(forecast_data['date'], forecast_data['forecast'],
                 color='red', linewidth=2.5, marker='s', markersize=4,
                 linestyle='--', label='Forecast', alpha=0.8, zorder=2)
        
        # Split line
        ax1.axvline(train_data['date'].max(), color='blue', 
                    linestyle='--', linewidth=2, alpha=0.5, label='Train/Test Split')
        
        # Get metrics for title
        state_results = results_df[results_df['State'] == state].iloc[0]
        
        ax1.set_title(f'{state} - ARIMAX Intervention Forecast\n' + 
                      f'MAE: {state_results["MAE"]:,.0f} | R²: {state_results["R2"]:.3f}',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Unemployment Claims', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Right plot: Errors
        ax2 = axes[idx, 1]
        
        ax2.plot(forecast_data['date'], forecast_data['error'],
                 color='purple', linewidth=2, marker='o', markersize=4)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(forecast_data['error'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {forecast_data["error"].mean():,.0f}')
        
        std_error = forecast_data['error'].std()
        ax2.fill_between(forecast_data['date'], -std_error, std_error,
                         alpha=0.2, color='gray', label=f'±1σ: {std_error:,.0f}')
        
        ax2.set_title(f'{state} - Forecast Errors',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Forecast Error', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
    except Exception as e:
        print(f" Could not plot {state}: {e}")

plt.tight_layout()
plt.savefig('Results/intervention/forecasts_visualization.png', dpi=300, bbox_inches='tight')
print("    Saved: Results/intervention/forecasts_visualization.png")
plt.close()

# ============================================
# VISUALIZATION: COEFFICIENT HEATMAP
# ============================================

if len(coefficient_list) > 0:
    print(f"\n{'='*80}")
    print("CREATING COEFFICIENT HEATMAP")
    print(f"{'='*80}")
    
    # Pivot for heatmap
    coef_pivot = intervention_coefficients.pivot(
        index='State',
        columns='Intervention',
        values='Coefficient'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    import seaborn as sns
    sns.heatmap(coef_pivot, annot=True, fmt='.0f', cmap='RdBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'Coefficient Value'})
    
    ax.set_title('Intervention Coefficients by State\n(Positive = Increased Claims, Negative = Decreased Claims)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Intervention Period', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('Results/intervention/coefficient_heatmap.png', dpi=300, bbox_inches='tight')
    print("    Saved: Results/intervention/coefficient_heatmap.png")
    plt.close()

# ============================================
# FINAL SUMMARY
# ============================================

print(f"\n{'='*80}")
print("INTERVENTION ANALYSIS COMPLETE")
print(f"{'='*80}")

print(f"\n Successfully analyzed {len(states)} states")
print(f" Modeled {len(interventions)} intervention periods")

print(f"\n FILES CREATED:")
print(f"    Results/intervention/arimax_intervention_results.csv")
print(f"    Results/intervention/intervention_coefficients.csv")
print(f"    Results/intervention/intervention_periods_visualization.png")
print(f"    Results/intervention/forecasts_visualization.png")
print(f"    Results/intervention/coefficient_heatmap.png")
for state in states:
    print(f"   Results/intervention/forecast_{state}.csv")

print(f"\n INTERPRETATION:")
print(f"   • Positive coefficients → Intervention increased claims")
print(f"   • Negative coefficients → Intervention decreased claims")
print(f"   • Check p-values for statistical significance")




