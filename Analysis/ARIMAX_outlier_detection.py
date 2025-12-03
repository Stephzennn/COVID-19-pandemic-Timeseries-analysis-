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

"""
ARIMAX with Outlier Detection Methods
Purpose: Test if handling outliers improves performance
Methods: 
  1. Outlier dummy variables
  2. Log transformation
  3. Winsorization
"""

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
# OUTLIER DETECTION
# ============================================

def detect_outliers(series, method='zscore', threshold=3):
    """
    Detect outliers using various methods
    
    Parameters:
    - method: 'zscore', 'iqr', or 'isolation'
    - threshold: for zscore (default 3), or IQR multiplier (default 1.5)
    """
    
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
        
    elif method == 'iqr':
        # Interquartile Range method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers

# ============================================
# VISUALIZE OUTLIERS
# ============================================
fig, axes = plt.subplots(len(states), 1, figsize=(14, 5*len(states)))

if len(states) == 1:
    axes = [axes]

for idx, state in enumerate(states):
    claims = train_data[f'{state}_claims']
    
    # Detect outliers
    outliers_zscore = detect_outliers(claims, method='zscore', threshold=3)
    outliers_iqr = detect_outliers(claims, method='iqr', threshold=1.5)
    
    # Plot
    ax = axes[idx]
    ax.plot(train_data['date'], claims, 
            linewidth=2, label='Claims', color='blue', alpha=0.7)
    
    # Mark outliers
    ax.scatter(train_data['date'][outliers_zscore], claims[outliers_zscore],
               color='red', s=100, marker='o', label='Outliers (Z-score)', zorder=5)
    
    ax.scatter(train_data['date'][outliers_iqr], claims[outliers_iqr],
               color='orange', s=50, marker='x', label='Outliers (IQR)', zorder=4)
    
    # Add mean and std bands
    mean_val = claims.mean()
    std_val = claims.std()
    ax.axhline(mean_val, color='green', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_val:,.0f}')
    ax.fill_between(train_data['date'], 
                     mean_val - 3*std_val, 
                     mean_val + 3*std_val,
                     alpha=0.2, color='gray', label='±3σ band')
    
    ax.set_title(f'{state} - Outlier Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Claims')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Print outlier info
    print(f"\n{state}:")
    print(f"  Z-score outliers: {outliers_zscore.sum()} ({outliers_zscore.sum()/len(claims)*100:.1f}%)")
    print(f"  IQR outliers: {outliers_iqr.sum()} ({outliers_iqr.sum()/len(claims)*100:.1f}%)")
    
    if outliers_zscore.sum() > 0:
        outlier_dates = train_data.loc[outliers_zscore, 'date']
        print(f"  Outlier dates: {outlier_dates.dt.strftime('%Y-%m-%d').tolist()}")

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
print("\n Saved: outlier_detection.png")
plt.close()

# ============================================
# Method 1: FIT ARIMAX FOR EACH STATE WITH OUTLIER HANDLING
# ============================================
print(f"\n{'='*80}")
print("METHOD 1: ARIMAX WITH OUTLIER INDICATOR VARIABLES")
print(f"{'='*80}")

results_outlier_dummies = {}

for state in states:
    print(f"\n{'='*60}")
    print(f"STATE: {state}")
    print(f"{'='*60}")
    
    # Get data
    y_train = train_data[f'{state}_claims']
    y_test = test_data[f'{state}_claims']
    
    # Detect outliers in training data
    outliers = detect_outliers(y_train, method='zscore', threshold=3)
    
    print(f"\nDetected {outliers.sum()} outliers in training data")
    
    # Create outlier dummy variable
    outlier_dummy_train = outliers.astype(int)
    outlier_dummy_test = pd.Series(0, index=range(len(test_data)))  # No outliers in test

    # Exogenous variables: COVID + outlier dummy
    X_train = pd.DataFrame({
        'cases': train_data[f'{state}_cases'],
        'deaths': train_data[f'{state}_deaths'],
        'outlier': outlier_dummy_train
    })
    
    X_test = pd.DataFrame({
        'cases': test_data[f'{state}_cases'],
        'deaths': test_data[f'{state}_deaths'],
        'outlier': outlier_dummy_test
    })
    
    try:
        # Fit model
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(2, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False, maxiter=200)

        # Forecast
        forecast = fitted.forecast(steps=len(test_data), exog=X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        r2 = r2_score(y_test, forecast)
        mape = np.mean(np.abs((y_test - forecast) / y_test.replace(0, np.nan))) * 100
        
        results_outlier_dummies[state] = {
            'model': fitted,
            'forecast': forecast,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'outlier_count': outliers.sum()
        }
        
        print(f"\n Model fitted with outlier dummies")
        print(f"  MAE: {mae:,.0f}")
        print(f"  RMSE: {rmse:,.0f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        
    except Exception as e:
        print(f" Failed: {e}")
        results_outlier_dummies[state] = None

# Save results
summary_outlier = []
for state in states:
    if results_outlier_dummies.get(state):
        r = results_outlier_dummies[state]
        summary_outlier.append({
            'State': state,
            'Model': 'ARIMAX_Outlier_Dummies',
            'Outliers_Detected': r['outlier_count'],
            'MAE': r['MAE'],
            'RMSE': r['RMSE'],
            'R2': r['R2'],
            'MAPE': r['MAPE']
        })

pd.DataFrame(summary_outlier).to_csv('arimax_outlier_dummies_results.csv', index=False)
print("\n Saved: arimax_outlier_dummies_results.csv")

# ============================================
# METHOD 2: LOG TRANSFORMATION
# ============================================

print(f"\n{'='*80}")
print("METHOD 2: LOG-TRANSFORMED ARIMAX")
print(f"{'='*80}")

results_log = {}

for state in states:
    print(f"\n{'='*60}")
    print(f"STATE: {state}")
    print(f"{'='*60}")

    # Get data
    y_train = train_data[f'{state}_claims']
    y_test = test_data[f'{state}_claims']
    
    X_train = train_data[[f'{state}_cases', f'{state}_deaths']]
    X_test = test_data[[f'{state}_cases', f'{state}_deaths']]
    
    # Log transform (add 1 to handle zeros)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)
    
    print(f"\nApplied log(1+x) transformation")

    try:
        # Fit model on log-transformed data
        model = SARIMAX(
            y_train_log,
            exog=X_train_log,
            order=(2, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False, maxiter=200)
        
        # Forecast (in log space)
        forecast_log = fitted.forecast(steps=len(test_data), exog=X_test_log)
        
        # Back-transform to original scale
        forecast = np.expm1(forecast_log)  # inverse of log1p
        
        # Metrics (on original scale)
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        r2 = r2_score(y_test, forecast)
        mape = np.mean(np.abs((y_test - forecast) / y_test.replace(0, np.nan))) * 100
        
        results_log[state] = {
            'model': fitted,
            'forecast': forecast,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        print(f"\n Log-transformed model fitted")
        print(f"  MAE: {mae:,.0f}")
        print(f"  RMSE: {rmse:,.0f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        
    except Exception as e:
        print(f" Failed: {e}")
        results_log[state] = None

# Save results
summary_log = []
for state in states:
    if results_log.get(state):
        r = results_log[state]
        summary_log.append({
            'State': state,
            'Model': 'ARIMAX_Log_Transform',
            'MAE': r['MAE'],
            'RMSE': r['RMSE'],
            'R2': r['R2'],
            'MAPE': r['MAPE']
        })
pd.DataFrame(summary_log).to_csv('arimax_log_transform_results.csv', index=False)
print("\n Saved: arimax_log_transform_results.csv")

print(f"\n{'='*80}")
print("OUTLIER DETECTION & ROBUST METHODS COMPLETE")
print(f"{'='*80}")