import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

"""
ACF/PACF Analysis for ARIMA Order Selection
Purpose: Determine optimal p, d, q parameters
Run this FIRST to guide other analyses
"""

# ============================================
# LOAD DATA
# ============================================

print("="*80)
print("ACF/PACF ANALYSIS FOR ARIMA ORDER SELECTION")
print("="*80)

# Load states
with open('states_list.txt', 'r') as f:
    states = f.read().strip().split(',')

# Load data
train_data = pd.read_csv('train_data.csv')
train_data['date'] = pd.to_datetime(train_data['date'])

print(f"\nAnalyzing {len(states)} states: {states}")

# ============================================
# STATIONARITY TESTS
# ============================================

def test_stationarity(series, name):
    """Test if series is stationary using ADF and KPSS tests"""
    
    print(f"\n{'='*60}")
    print(f"STATIONARITY TESTS: {name}")
    print(f"{'='*60}")
    
    # ADF Test (null hypothesis: non-stationary)
    adf_result = adfuller(series.dropna())
    print(f"\nAugmented Dickey-Fuller Test:")
    print(f"  Test Statistic: {adf_result[0]:.4f}")
    print(f"  P-value: {adf_result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if adf_result[1] < 0.05:
        print(f"  ✓ Series is stationary (reject H0, p < 0.05)")
        adf_stationary = True
    else:
        print(f"  ✗ Series is non-stationary (fail to reject H0, p >= 0.05)")
        adf_stationary = False
    
    # KPSS Test (null hypothesis: stationary)
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"\nKPSS Test:")
    print(f"  Test Statistic: {kpss_result[0]:.4f}")
    print(f"  P-value: {kpss_result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"    {key}: {value:.4f}")
    
    if kpss_result[1] >= 0.05:
        print(f"  ✓ Series is stationary (fail to reject H0, p >= 0.05)")
        kpss_stationary = True
    else:
        print(f"  ✗ Series is non-stationary (reject H0, p < 0.05)")
        kpss_stationary = False
    
    # Interpretation
    print(f"\nINTERPRETATION:")
    if adf_stationary and kpss_stationary:
        print(f"  ✓ Both tests agree: Series is STATIONARY")
        print(f"  → Differencing parameter d = 0")
        return 0
    elif not adf_stationary and not kpss_stationary:
        print(f"  ✓ Both tests agree: Series is NON-STATIONARY")
        print(f"  → Need differencing, d = 1")
        return 1
    else:
        print(f"  ⚠ Tests disagree - inconclusive")
        print(f"  → Try d = 1 (safe default)")
        return 1

# ============================================
# ACF/PACF ANALYSIS FUNCTION
# ============================================

def analyze_acf_pacf(series, state, series_type="Original"):
    """
    Create comprehensive ACF/PACF plots and provide interpretation
    """
    
    # Clean data
    series_clean = series.dropna()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{state} - {series_type} Series ACF/PACF Analysis', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # Plot 1: Time series
    axes[0, 0].plot(series_clean.index, series_clean.values, linewidth=1.5)
    axes[0, 0].set_title(f'{series_type} Series', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add mean line
    mean_val = series_clean.mean()
    axes[0, 0].axhline(mean_val, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_val:.0f}')
    axes[0, 0].legend()
    
    # Plot 2: Histogram
    axes[0, 1].hist(series_clean, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(mean_val, color='red', linestyle='--', linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ACF
    plot_acf(series_clean, lags=40, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('Autocorrelation Function (ACF)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    
    # Plot 4: PACF
    plot_pacf(series_clean, lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('PACF')
    
    plt.tight_layout()
    filename = f'acf_pacf_{state}_{series_type.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # Interpretation
    return interpret_acf_pacf(series_clean, state, series_type)

def interpret_acf_pacf(series, state, series_type):
    """
    Interpret ACF/PACF patterns and suggest ARIMA orders
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    print(f"\n{'='*60}")
    print(f"ACF/PACF INTERPRETATION: {state} - {series_type}")
    print(f"{'='*60}")
    
    # Calculate ACF and PACF
    acf_values = acf(series, nlags=40, fft=False)
    pacf_values = pacf(series, nlags=40)
    
    # Find significant lags (beyond 95% CI)
    n = len(series)
    conf_interval = 1.96 / np.sqrt(n)
    
    # ACF significant lags
    acf_sig = np.where(np.abs(acf_values[1:]) > conf_interval)[0] + 1
    
    # PACF significant lags
    pacf_sig = np.where(np.abs(pacf_values[1:]) > conf_interval)[0] + 1
    
    print(f"\nACF Analysis:")
    print(f"  Significant lags: {acf_sig[:10] if len(acf_sig) > 0 else 'None'}")
    
    # Check for slow decay (non-stationarity indicator)
    if len(acf_sig) > 20:
        print(f"  ⚠ Many significant lags → possible non-stationarity")
        print(f"  → Series may need differencing (d=1)")
        acf_pattern = "slow_decay"
    elif len(acf_sig) > 0 and acf_sig[0] <= 5:
        print(f"  ✓ ACF cuts off at lag {acf_sig[-1]} → MA process")
        print(f"  → Suggests q = {min(acf_sig[-1], 3)}")
        acf_pattern = "cutoff"
    else:
        print(f"  ✓ ACF decays quickly → stationary")
        acf_pattern = "quick_decay"
    
    print(f"\nPACF Analysis:")
    print(f"  Significant lags: {pacf_sig[:10] if len(pacf_sig) > 0 else 'None'}")
    
    if len(pacf_sig) > 0 and pacf_sig[0] <= 5:
        print(f"  ✓ PACF cuts off at lag {pacf_sig[-1]} → AR process")
        print(f"  → Suggests p = {min(pacf_sig[-1], 3)}")
        pacf_pattern = "cutoff"
    else:
        print(f"  ✓ PACF decays gradually")
        pacf_pattern = "decay"
    
    # Suggest ARIMA order
    print(f"\nSUGGESTED ARIMA ORDER:")
    
    if acf_pattern == "slow_decay":
        print(f"  ⚠ Need differencing first!")
        print(f"  → Analyze differenced series to determine p and q")
        suggested_p, suggested_q = None, None
    else:
        # Suggest p from PACF
        if pacf_pattern == "cutoff" and len(pacf_sig) > 0:
            suggested_p = min(pacf_sig[-1], 3)
        else:
            suggested_p = 2  # default
        
        # Suggest q from ACF
        if acf_pattern == "cutoff" and len(acf_sig) > 0:
            suggested_q = min(acf_sig[-1], 3)
        else:
            suggested_q = 2  # default
        
        print(f"  p (AR order) = {suggested_p}")
        print(f"  q (MA order) = {suggested_q}")
    
    # Check for seasonality
    seasonal_lags = [52, 104]  # Weekly data: 52 weeks = 1 year
    seasonal_acf = [acf_values[lag] if lag < len(acf_values) else 0 
                    for lag in seasonal_lags]
    
    print(f"\nSEASONALITY CHECK:")
    print(f"  ACF at lag 52: {seasonal_acf[0]:.3f}")
    if len(seasonal_acf) > 1:
        print(f"  ACF at lag 104: {seasonal_acf[1]:.3f}")
    
    if abs(seasonal_acf[0]) > conf_interval:
        print(f"  ⚠ Significant seasonal component detected!")
        print(f"  → Consider seasonal ARIMA with s=52")
        has_seasonality = True
    else:
        print(f"  ✓ No strong seasonal component")
        has_seasonality = False
    
    return {
        'p': suggested_p,
        'q': suggested_q,
        'seasonality': has_seasonality}

