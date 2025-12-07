import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import itertools

# ============================================
# LOAD PROCESSED DATA
# ============================================

print("="*80)
print("ARIMAX MODEL - SINGLE STATE WITH COVID FEATURES")
print("="*80)

# Load states list
with open('states_list.txt', 'r') as f:
    states = f.read().strip().split(',')

print(f"\n Detected states: {states}")

# Load training and test data (wide format)
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"\nTrain data: {len(train_data)} weeks")
print(f"Test data: {len(test_data)} weeks")
print(f"Train date range: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Test date range: {test_data['date'].min()} to {test_data['date'].max()}")

# ============================================
# ARIMAX CONFIGURATION - GRID SEARCH with CV
# ============================================

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import itertools

def evaluate_arimax_model(y, exog, order, seasonal_order):
    """Evaluate ARIMAX model using time series cross-validation"""
    try:
        model = SARIMAX(y, exog=exog, order=order, 
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=100)
        return fitted.aic, fitted.bic
    except:
        return np.inf, np.inf

# Grid search
p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)
s = 52

best_aic = np.inf
best_order = None
best_seasonal = None

print("Grid searching for optimal order...")

for state in states:
    y = train_data[f'{state}_claims']
    X = train_data[[f'{state}_cases', f'{state}_deaths']]
    
    print(f"\nSearching for {state}...")
    
    
    orders_to_try = [
        ((1,1,1), (0,0,0,0)),  # Simple, no seasonal
        ((1,1,1), (1,0,1,52)), # Simple with seasonal
        ((2,1,2), (0,0,0,0)),  # Medium, no seasonal
        ((2,1,2), (1,0,1,52)), 
        ((3,1,3), (1,0,1,52)), # Complex
        ((1,1,0), (0,0,0,0)),  # AR only
        ((0,1,1), (0,0,0,0)),  # MA only
        ((2,1,0), (1,0,0,52)), # AR with seasonal AR
    ]
    
    for order, seasonal_order in orders_to_try:
        aic, bic = evaluate_arimax_model(y, X, order, seasonal_order)
        print(f"  {order} {seasonal_order}: AIC={aic:.0f}, BIC={bic:.0f}")
        
        if aic < best_aic:
            best_aic = aic
            best_order = order
            best_seasonal = seasonal_order
    
    print(f"\n Best for {state}: {best_order} {best_seasonal} (AIC: {best_aic:.0f})")

ARIMAX_CONFIG = {
    'order': best_order,
    'seasonal_order': best_seasonal,
    'enforce_stationarity': False,
    'enforce_invertibility': False,
    'maxiter': 200,
    'disp': False
}

print(f"\nARIMAX Configuration:")
print(f"  Order (p,d,q): {ARIMAX_CONFIG['order']}")
print(f"  Seasonal Order (P,D,Q,s): {ARIMAX_CONFIG['seasonal_order']}")

# ============================================
# FIT ARIMAX FOR EACH STATE
# ============================================

print(f"\n{'='*80}")
print("FITTING ARIMAX MODELS")
print(f"{'='*80}")

arimax_results = {}

for state in states:
    print(f"\n{'='*60}")
    print(f"STATE: {state}")
    print(f"{'='*60}")
    
    # Extract target variable (claims)
    y_train = train_data[f'{state}_claims'].copy()
    y_test = test_data[f'{state}_claims'].copy()
    
    # Extract exogenous variables (COVID features)
    X_train = train_data[[f'{state}_cases', f'{state}_deaths']].copy()
    X_test = test_data[[f'{state}_cases', f'{state}_deaths']].copy()
    
    print(f"\nTarget variable: {state}_claims")
    print(f"  Train size: {len(y_train)}")
    print(f"  Test size: {len(y_test)}")
    
    print(f"\nExogenous variables: {X_train.columns.tolist()}")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    
    # FIXED: Check for missing values correctly (both lines)
    train_missing = y_train.isnull().sum() + X_train.isnull().sum().sum()
    if train_missing > 0:
        print(f"\n WARNING: {train_missing} missing values in training data")
        print(f"  y_train missing: {y_train.isnull().sum()}")
        print(f"  X_train missing:\n{X_train.isnull().sum()}")
        print("Filling with forward fill...")
        y_train = y_train.fillna(method='ffill')
        X_train = X_train.fillna(method='ffill')
    
    test_missing = y_test.isnull().sum() + X_test.isnull().sum().sum()
    if test_missing > 0:
        print(f"\n WARNING: {test_missing} missing values in test data")
        print(f"  y_test missing: {y_test.isnull().sum()}")
        print(f"  X_test missing:\n{X_test.isnull().sum()}")
        print("Filling with forward fill...")
        y_test = y_test.fillna(method='ffill')
        X_test = X_test.fillna(method='ffill')
    
    try:
        print(f"\nFitting ARIMAX model...")
            
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=ARIMAX_CONFIG['order'],
            seasonal_order=ARIMAX_CONFIG['seasonal_order'],
            enforce_stationarity=ARIMAX_CONFIG['enforce_stationarity'],
            enforce_invertibility=ARIMAX_CONFIG['enforce_invertibility']
        )
        
        fitted_model = model.fit(
            disp=ARIMAX_CONFIG['disp'],
            maxiter=ARIMAX_CONFIG['maxiter']
        )
        
        print(f" Model fitted successfully")
        
        # Make predictions
        print(f"\nGenerating forecasts...")
        forecast = fitted_model.forecast(steps=len(test_data), exog=X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        
        mape = np.mean(np.abs((y_test.values - forecast.values) / y_test.values)) * 100
        r2 = r2_score(y_test, forecast)
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(y_test.values))
        pred_direction = np.sign(np.diff(forecast.values))
        direction_acc = np.mean(actual_direction == pred_direction) * 100
        
        # Store results
        arimax_results[state] = {
            'model': fitted_model,
            'forecast': forecast,
            'actual': y_test,
            'train_dates': train_data['date'],
            'test_dates': test_data['date'],
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Direction_Acc': direction_acc,
            'AIC': fitted_model.aic,
            'BIC': fitted_model.bic
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {state}")
        print(f"{'='*60}")
        print(f"  MAE:                {mae:,.0f}")
        print(f"  RMSE:               {rmse:,.0f}")
        print(f"  MAPE:               {smape:.2f}%")
        print(f"  R²:                 {r2:.3f}")
        print(f"  Direction Accuracy: {direction_acc:.1f}%")
        print(f"  AIC:                {fitted_model.aic:.0f}")
        print(f"  BIC:                {fitted_model.bic:.0f}")
        
        print(f"\n  Converged: {fitted_model.mle_retvals['converged']}")
        
    except Exception as e:
        print(f"\n ERROR: ARIMAX failed for {state}")
        print(f"   {str(e)}")
        print(f"\n   Trying simpler model specification...")
        
        try:
            model_simple = SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model_simple.fit(disp=False, maxiter=100)
            forecast = fitted_model.forecast(steps=len(test_data), exog=X_test)
            
            mae = mean_absolute_error(y_test, forecast)
            rmse = np.sqrt(mean_squared_error(y_test, forecast))
            
            # Replace MAPE calculation with SMAPE
            mape = np.mean(np.abs((y_test.values - forecast.values) / y_test.values)) * 100
            r2 = r2_score(y_test, forecast)
            
            actual_direction = np.sign(np.diff(y_test.values))
            pred_direction = np.sign(np.diff(forecast.values))
            direction_acc = np.mean(actual_direction == pred_direction) * 100
            
            arimax_results[state] = {
                'model': fitted_model,
                'forecast': forecast,
                'actual': y_test,
                'train_dates': train_data['date'],
                'test_dates': test_data['date'],
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'Direction_Acc': direction_acc,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'note': 'ARIMAX model (3,1,3) with no seasonality'
            }
            
            print(f" Simpler model fitted successfully")
            print(f"  MAE: {mae:,.0f}, RMSE: {rmse:,.0f}, R²: {r2:.3f}")
            
        except Exception as e2:
            print(f"   Simpler model also failed: {str(e2)}")
            arimax_results[state] = None

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

summary_data = []

for state in states:
    if arimax_results.get(state):
        result = arimax_results[state]
        summary_data.append({
            'State': state,
            'Model': 'ARIMAX',
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'MAPE': result['MAPE'],
            'R2': result['R2'],
            'Direction_Acc': result['Direction_Acc'],
            'AIC': result['AIC'],
            'BIC': result['BIC'],
            'Note': result.get('note', 'Full model')
        })

summary_df = pd.DataFrame(summary_data)

print("\n" + "="*80)
print("ARIMAX SUMMARY - ALL STATES")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

summary_df.to_csv('arimax_results.csv', index=False)
print(f"\n Saved 'arimax_results.csv'")

# Save individual forecasts
for state in states:
    if arimax_results.get(state):
        forecast_df = pd.DataFrame({
            'date': test_data['date'],
            'actual': arimax_results[state]['actual'].values,
            'forecast': arimax_results[state]['forecast'].values,
            'error': arimax_results[state]['actual'].values - arimax_results[state]['forecast'].values
        })
        forecast_df.to_csv(f'arimax_forecast_{state}.csv', index=False)
        print(f" Saved 'arimax_forecast_{state}.csv'")

# ============================================
# VISUALIZATION
# ============================================

print(f"\n{'='*80}")
print("VISUALIZATIONS")
print(f"{'='*80}")

n_states = len([s for s in states if arimax_results.get(s)])

if n_states > 0:
    fig, axes = plt.subplots(n_states, 2, figsize=(16, 5*n_states))

    if n_states == 1:
        axes = axes.reshape(1, -1)

    plot_idx = 0

    for state in states:
        if not arimax_results.get(state):
            continue
        
        result = arimax_results[state]
        
        # Left: Actual vs Forecast
        ax1 = axes[plot_idx, 0]
        
        ax1.plot(result['train_dates'], train_data[f'{state}_claims'],
                 color='gray', alpha=0.3, linewidth=1, label='Training Data')
        
        ax1.plot(result['test_dates'], result['actual'],
                 color='lightblue', linewidth=2.5, marker='o', markersize=5,
                 label='Actual', zorder=3)
        
        ax1.plot(result['test_dates'], result['forecast'],
                 color='red', linewidth=2.5, marker='s', markersize=4,
                 linestyle='--', label='ARIMAX Forecast', alpha=0.8, zorder=2)
        
        ax1.axvline(result['train_dates'].max(), color='blue', 
                    linestyle='--', linewidth=2, alpha=0.5, label='Train/Test Split')
        
        ax1.set_title(f'{state} - ARIMAX Forecast vs Actual\n' + 
                      f'MAE: {result["MAE"]:,.0f} | RMSE: {result["RMSE"]:,.0f} | R²: {result["R2"]:.3f}',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Unemployment Claims (thousands)', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Right: Errors
        ax2 = axes[plot_idx, 1]
        
        errors = result['actual'].values - result['forecast'].values
        
        ax2.plot(result['test_dates'], errors, 
                 color='purple', linewidth=2, marker='o', markersize=4)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(errors.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean Error: {errors.mean():,.0f}')
        
        std_error = errors.std()
        ax2.fill_between(result['test_dates'], -std_error, std_error,
                         alpha=0.2, color='gray', label=f'±1 Std: {std_error:,.0f}')
        
        ax2.set_title(f'{state} - Forecast Errors\n' + 
                      f'Mean: {errors.mean():,.0f} | Std: {errors.std():,.0f}',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Forecast Error', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plot_idx += 1

    plt.tight_layout()
    plt.savefig('arimax_forecasts.png', dpi=300, bbox_inches='tight')
    print(f" Saved 'arimax_forecasts.png'")
    plt.close()

# ============================================
# RESIDUAL DIAGNOSTICS
# ============================================

print(f"\n{'='*80}")
print("RESIDUAL DIAGNOSTICS")
print(f"{'='*80}")

if n_states > 0:
    fig, axes = plt.subplots(n_states, 2, figsize=(16, 5*n_states))

    if n_states == 1:
        axes = axes.reshape(1, -1)

    plot_idx = 0

    for state in states:
        if not arimax_results.get(state):
            continue
        
        result = arimax_results[state]
        residuals = result['model'].resid
        
        # Histogram
        ax1 = axes[plot_idx, 0]
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_title(f'{state} - Residual Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Residuals')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        ax2 = axes[plot_idx, 1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'{state} - Q-Q Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plot_idx += 1

    plt.tight_layout()
    plt.savefig('arimax_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f" Saved 'arimax_diagnostics.png'")
    plt.close()

# ============================================
# FINAL SUMMARY
# ============================================

print(f"\n{'='*80}")
print("ARIMAX MODELING COMPLETE")
print(f"{'='*80}")

successful_states = len([s for s in states if arimax_results.get(s)])
print(f"\n Successfully modeled {successful_states} / {len(states)} states")

if successful_states > 0:
    print(f"\n PERFORMANCE SUMMARY:")
    for state in states:
        if arimax_results.get(state):
            result = arimax_results[state]
            print(f"\n  {state}:")
            print(f"    MAE:  {result['MAE']:>10,.0f}")
            print(f"    RMSE: {result['RMSE']:>10,.0f}")
            print(f"    R²:   {result['R2']:>10.3f}")
            print(f"    MAPE: {result['MAPE']:>10.2f}%")

    print(f"\n FILES CREATED:")
    print(f"    arimax_results.csv")
    for state in states:
        if arimax_results.get(state):
            print(f"    arimax_forecast_{state}.csv")
    print(f"    arimax_forecasts.png")
    print(f"    arimax_diagnostics.png")
