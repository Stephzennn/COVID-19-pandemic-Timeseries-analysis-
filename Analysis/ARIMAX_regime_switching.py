import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ARIMAX WITH MARKOV REGIME-SWITCHING")
print("="*80)

# ============================================
# LOAD DATA
# ============================================

print("\nLoading data...")

# Load data files
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Load states
with open('states_list.txt', 'r') as f:
    states = f.read().strip().split(',')

print(f"States: {states}")
print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Test period: {test_data['date'].min()} to {test_data['date'].max()}")
print(f"Training weeks: {len(train_data)}")
print(f"Test weeks: {len(test_data)}")

# ============================================
# MARKOV SWITCHING REGRESSION
# ============================================

markov_results = {}

for state in states:
    print(f"\n{'='*80}")
    print(f"MARKOV SWITCHING REGIME MODEL - {state}")
    print(f"{'='*80}")
    
    # Get training data
    y_train = train_data[f'{state}_claims']
    X_train = train_data[[f'{state}_cases', f'{state}_deaths']]
    
    print(f"\nTraining data:")
    print(f"  Claims: {y_train.min():,.0f} to {y_train.max():,.0f}")
    print(f"  Observations: {len(y_train)}")
    
    # ============================================
    # METHOD 1: Markov Switching Regression (2 Regimes)
    # ============================================
    
    print(f"\n{'-'*60}")
    print("METHOD 1: Markov Switching Regression")
    print(f"{'-'*60}")
    
    try:
        print("\nFitting Markov Switching model (2 regimes)...")
    
        
        # Fit Markov-switching regression
        model_ms = MarkovRegression(
            y_train,
            k_regimes=2,  # 2 regimes (pandemic vs endemic-like)
            exog=X_train,
            switching_variance=True,  # Allow different variances per regime
            switching_exog=False  # Keep COVID coefficients same (you can change this)
        )
        
        fitted_ms = model_ms.fit(
            maxiter=1000,
            disp=False,
            method='bfgs'
        )
        
        print(f" Model converged: {fitted_ms.mle_retvals.get('converged', 'Unknown')}")
        
        # ============================================
        # ANALYZE REGIMES
        # ============================================
        
        print(f"\n{'-'*60}")
        print("REGIME ANALYSIS")
        print(f"{'-'*60}")
        
        # Get regime probabilities
        regime_probs = fitted_ms.smoothed_marginal_probabilities
        
        # Regime 0 vs Regime 1
        regime_0_prob = regime_probs[0].mean()
        regime_1_prob = regime_probs[1].mean()
        
        print(f"\nAverage regime probabilities:")
        print(f"  Regime 0: {regime_0_prob:.1%}")
        print(f"  Regime 1: {regime_1_prob:.1%}")
        
        # Identify dominant regime in each period
        dominant_regime = regime_probs.idxmax(axis=1)
        
        print(f"\nRegime distribution over time:")
        print(f"  Regime 0: {(dominant_regime == 0).sum()} weeks")
        print(f"  Regime 1: {(dominant_regime == 1).sum()} weeks")
        
        # Show when regime switches occurred
        regime_changes = dominant_regime.diff()
        switches = regime_changes[regime_changes != 0]
        
        print(f"\nRegime switches detected: {len(switches)}")
        if len(switches) > 0:
            print(f"First switch: {train_data.iloc[switches.index[0]]['date']}")
            print(f"Last switch: {train_data.iloc[switches.index[-1]]['date']}")
        
        # Regime parameters
        print(f"\n{'-'*60}")
        print("REGIME PARAMETERS")
        print(f"{'-'*60}")
        
        params = fitted_ms.params
        
        # Try to get regime-specific intercepts
        try:
            const_0 = params['const[0]'] if 'const[0]' in params.index else params.get('const.regime[0]', None)
            const_1 = params['const[1]'] if 'const[1]' in params.index else params.get('const.regime[1]', None)
    
            if const_0 is not None:
                print(f"\nRegime 0 (Intercept): {const_0:,.2f}")
            if const_1 is not None:
                print(f"Regime 1 (Intercept): {const_1:,.2f}")
        except Exception as e:
            print(f" Could not extract regime intercepts: {e}")
            const_0 = None
            const_1 = None
            
        # COVID coefficients
        try:
            # Check which parameter names exist
            param_names = params.index.tolist()
    
            print(f"\nAvailable parameters: {param_names[:5]}...")  # Show first 5
    
            # Try different possible names
            cases_coef = None
            deaths_coef = None

            for name in param_names:
                if 'cases' in name.lower():
                    cases_coef = params[name]
                    print(f"\nCOVID Cases coefficient ({name}): {cases_coef:.6f}")
                if 'deaths' in name.lower():
                    deaths_coef = params[name]
                    print(f"COVID Deaths coefficient ({name}): {deaths_coef:.6f}")

            if cases_coef is None:
                print(f"\n Could not find cases coefficient")
            if deaths_coef is None:
                print(f" Could not find deaths coefficient")

        except Exception as e:
            print(f" Could not extract COVID coefficients: {e}")
            cases_coef = None
            deaths_coef = None

        # Variance parameters
        try:
            sigma0 = params['sigma2[0]'] if 'sigma2[0]' in params.index else params.get('sigma2.regime[0]', None)
            sigma1 = params['sigma2[1]'] if 'sigma2[1]' in params.index else params.get('sigma2.regime[1]', None)
    
            if sigma0 is not None:
                print(f"\nRegime 0 variance: {sigma0:,.2f}")
            if sigma1 is not None:
                print(f"Regime 1 variance: {sigma1:,.2f}")
        except Exception as e:
            print(f" Could not extract variance parameters: {e}")
            sigma0, sigma1 = None, None
        
        # Transition probabilities
        print(f"\n{'-'*60}")
        print("TRANSITION PROBABILITIES")
        print(f"{'-'*60}")

        # Get transition matrix from the model
        try:
            # Try to get from fitted model's transition matrix
            trans_matrix = fitted_ms.regime_transition
    
            p00 = trans_matrix[0, 0]
            p01 = trans_matrix[0, 1]
            p10 = trans_matrix[1, 0]
            p11 = trans_matrix[1, 1]
    
            print(f"P(Regime 0 → Regime 0): {p00:.3f}")
            print(f"P(Regime 0 → Regime 1): {p01:.3f}")
            print(f"P(Regime 1 → Regime 0): {p10:.3f}")
            print(f"P(Regime 1 → Regime 1): {p11:.3f}")
    
            # Expected duration in each regime
            if p00 < 1.0:
                expected_duration_0 = 1 / (1 - p00)
            else:
                expected_duration_0 = np.inf
    
            if p11 < 1.0:
                expected_duration_1 = 1 / (1 - p11)
            else:
                expected_duration_1 = np.inf
    
            print(f"\nExpected regime duration:")
            if np.isfinite(expected_duration_0):
                print(f"  Regime 0: {expected_duration_0:.1f} weeks")
            else:
                print(f"  Regime 0: Always (no switching)")
    
            if np.isfinite(expected_duration_1):
                print(f"  Regime 1: {expected_duration_1:.1f} weeks")
            else:
                print(f"  Regime 1: Always (no switching)")
    
        except Exception as e:
            print(f" Could not extract transition probabilities: {e}")
            print("Using estimated values from regime distribution...")
    
            # Fallback: estimate from observed regime changes
            dominant_regime = regime_probs.idxmax(axis=1)
            regime_changes = dominant_regime.diff()
    
            # Count transitions
        n_00 = ((dominant_regime == 0) & (dominant_regime.shift(-1) == 0)).sum()
        n_01 = ((dominant_regime == 0) & (dominant_regime.shift(-1) == 1)).sum()
        n_10 = ((dominant_regime == 1) & (dominant_regime.shift(-1) == 0)).sum()
        n_11 = ((dominant_regime == 1) & (dominant_regime.shift(-1) == 1)).sum()
    
            # Calculate probabilities
        if (n_00 + n_01) > 0:
            p00 = n_00 / (n_00 + n_01)
            p01 = n_01 / (n_00 + n_01)
        else:
            p00, p01 = 0.5, 0.5
    
        if (n_10 + n_11) > 0:
            p10 = n_10 / (n_10 + n_11)
            p11 = n_11 / (n_10 + n_11)
        else:
            p10, p11 = 0.5, 0.5
    
        print(f"Estimated transition probabilities:")
        print(f"P(Regime 0 → Regime 0): {p00:.3f}")
        print(f"P(Regime 0 → Regime 1): {p01:.3f}")
        print(f"P(Regime 1 → Regime 0): {p10:.3f}")
        print(f"P(Regime 1 → Regime 1): {p11:.3f}")
    
        expected_duration_0 = 1 / (1 - p00) if p00 < 1.0 else np.inf
        expected_duration_1 = 1 / (1 - p11) if p11 < 1.0 else np.inf
    
        print(f"\nExpected regime duration:")
        print(f"  Regime 0: {expected_duration_0:.1f} weeks")
        print(f"  Regime 1: {expected_duration_1:.1f} weeks")
        
        # ============================================
        # IN-SAMPLE EVALUATION
        # ============================================
        
        print(f"\n{'-'*60}")
        print("IN-SAMPLE PERFORMANCE")
        print(f"{'-'*60}")
        
        # Fitted values
        fitted_values = fitted_ms.fittedvalues
        
        # Calculate metrics
        insample_mae = mean_absolute_error(y_train, fitted_values)
        insample_rmse = np.sqrt(mean_squared_error(y_train, fitted_values))
        insample_r2 = r2_score(y_train, fitted_values)
        
        print(f"\nIn-sample metrics:")
        print(f"  MAE:  {insample_mae:>10,.0f}")
        print(f"  RMSE: {insample_rmse:>10,.0f}")
        print(f"  R²:   {insample_r2:>10.3f}")
        
        # ============================================
        # OUT-OF-SAMPLE FORECASTING (SIMPLIFIED)
        # ============================================
        
        print(f"\n{'-'*60}")
        print("OUT-OF-SAMPLE FORECASTING")
        print(f"{'-'*60}")
        
        # Get test data
        y_test = test_data[f'{state}_claims']
        X_test = test_data[[f'{state}_cases', f'{state}_deaths']]
        
        # Determine most likely regime in recent training data
        recent_regime_probs = regime_probs.iloc[-10:].mean()
        recent_regime = recent_regime_probs.idxmax()
        
        print(f"\nRecent regime probabilities (last 10 weeks):")
        print(f"  Regime 0: {recent_regime_probs[0]:.1%}")
        print(f"  Regime 1: {recent_regime_probs[1]:.1%}")
        print(f"  → Dominant regime: {recent_regime}")
                
        # Extract regime-specific intercepts
        const_0 = params['const[0]']
        const_1 = params['const[1]']
        
        # Use dominant regime's parameters
        if recent_regime == 0:
            forecast_const = const_0
            print(f"\nUsing Regime 0 parameters for forecasting")
        else:
            forecast_const = const_1
            print(f"\nUsing Regime 1 parameters for forecasting")
        
        # Get COVID coefficients
        beta_cases = params[f'{state}_cases'] if f'{state}_cases' in params.index else 0
        beta_deaths = params[f'{state}_deaths'] if f'{state}_deaths' in params.index else 0
        
        # Simple linear prediction using regime parameters
        forecast_ms = (forecast_const + 
                      beta_cases * X_test[f'{state}_cases'].values + 
                      beta_deaths * X_test[f'{state}_deaths'].values)
        
        # Calculate metrics
        mae_ms = mean_absolute_error(y_test, forecast_ms)
        rmse_ms = np.sqrt(mean_squared_error(y_test, forecast_ms))
        r2_ms = r2_score(y_test, forecast_ms)
        mape_ms = np.mean(np.abs((y_test.values - forecast_ms) / y_test.values)) * 100
        
        print(f"\nOut-of-sample metrics (Markov Switching):")
        print(f"  MAE:  {mae_ms:>10,.0f}")
        print(f"  RMSE: {rmse_ms:>10,.0f}")
        print(f"  R²:   {r2_ms:>10.3f}")
        print(f"  MAPE: {mape_ms:>10.2f}%")
        
        # ============================================
        # COMPARE WITH STANDARD ARIMAX
        # ============================================
        
        print(f"\n{'-'*60}")
        print("COMPARISON: Markov Switching vs Standard ARIMAX")
        print(f"{'-'*60}")
        
        # Fit standard ARIMAX for comparison
        print("\nFitting standard ARIMAX (1,1,1)...")
        
        model_arimax = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_arimax = model_arimax.fit(disp=False, maxiter=200)
        forecast_arimax = fitted_arimax.forecast(steps=len(test_data), exog=X_test)
        
        # ARIMAX metrics
        mae_arimax = mean_absolute_error(y_test, forecast_arimax)
        rmse_arimax = np.sqrt(mean_squared_error(y_test, forecast_arimax))
        r2_arimax = r2_score(y_test, forecast_arimax)
        mape_arimax = np.mean(np.abs((y_test.values - forecast_arimax) / y_test.values)) * 100
        
        print(f"\nOut-of-sample metrics (Standard ARIMAX):")
        print(f"  MAE:  {mae_arimax:>10,.0f}")
        print(f"  RMSE: {rmse_arimax:>10,.0f}")
        print(f"  R²:   {r2_arimax:>10.3f}")
        print(f"  MAPE: {mape_arimax:>10.2f}%")
        
        # Calculate improvement
        print(f"\n{'-'*60}")
        print("IMPROVEMENT FROM MARKOV SWITCHING")
        print(f"{'-'*60}")
        
        mae_improvement = ((mae_arimax - mae_ms) / mae_arimax) * 100
        r2_improvement = r2_ms - r2_arimax
        
        print(f"\nMAE improvement: {mae_improvement:+.1f}%")
        print(f"R² improvement: {r2_improvement:+.3f}")
        
        if mae_improvement > 0:
            print(f"✓ Markov Switching reduces error")
        else:
            print(f"✗ Standard ARIMAX performs better")
        
        # ============================================
        # STORE RESULTS
        # ============================================
        
        markov_results[state] = {
            'model': fitted_ms,
            'regime_probs': regime_probs,
            'dominant_regime': dominant_regime,
            'forecast_ms': forecast_ms,
            'forecast_arimax': forecast_arimax,
            'actual': y_test,
            'test_dates': test_data['date'],
            'metrics_ms': {
                'MAE': mae_ms,
                'RMSE': rmse_ms,
                'R2': r2_ms,
                'MAPE': mape_ms
            },
            'metrics_arimax': {
                'MAE': mae_arimax,
                'RMSE': rmse_arimax,
                'R2': r2_arimax,
                'MAPE': mape_arimax
            },
            'improvement': {
                'MAE_pct': mae_improvement,
                'R2_diff': r2_improvement
            },
            'regime_params': {
                'const_0': const_0,
                'const_1': const_1,
                'p00': p00,
                'p11': p11,
                'expected_duration_0': expected_duration_0,
                'expected_duration_1': expected_duration_1
            },
            'aic': fitted_ms.aic,
            'bic': fitted_ms.bic
        }
        
        print(f"\n {state} completed successfully")
        
    except Exception as e:
        print(f"\n Markov Switching failed for {state}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        markov_results[state] = {
            'error': str(e),
            'status': 'failed'
        }

# ============================================
# SUMMARY COMPARISON TABLE
# ============================================

print(f"\n{'='*80}")
print("SUMMARY: MARKOV SWITCHING VS STANDARD ARIMAX")
print(f"{'='*80}")

summary_data = []

for state in states:
    if state in markov_results and 'metrics_ms' in markov_results[state]:
        result = markov_results[state]
        
        summary_data.append({
            'State': state,
            'MS_R2': result['metrics_ms']['R2'],
            'ARIMAX_R2': result['metrics_arimax']['R2'],
            'MS_MAE': result['metrics_ms']['MAE'],
            'ARIMAX_MAE': result['metrics_arimax']['MAE'],
            'MAE_Improvement_%': result['improvement']['MAE_pct'],
            'R2_Improvement': result['improvement']['R2_diff']
        })

if len(summary_data) > 0:
    summary_df = pd.DataFrame(summary_data)
    
    print("\nPerformance Comparison:")
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv('Markov_Switching_Comparison.csv', index=False)
    print(f"\n✓ Saved: Markov_Switching_Comparison.csv")
    
    # Overall statistics
    print(f"\n{'-'*60}")
    print("OVERALL STATISTICS")
    print(f"{'-'*60}")
    
    avg_improvement_mae = summary_df['MAE_Improvement_%'].mean()
    avg_improvement_r2 = summary_df['R2_Improvement'].mean()
    
    print(f"\nAverage MAE improvement: {avg_improvement_mae:+.1f}%")
    print(f"Average R² improvement: {avg_improvement_r2:+.3f}")
    
    states_improved = (summary_df['MAE_Improvement_%'] > 0).sum()
    print(f"\nStates with improved performance: {states_improved}/{len(states)}")

# ============================================
# VISUALIZATION
# ============================================

print(f"\n{'-'*60}")
print("CREATING VISUALIZATIONS")
print(f"{'-'*60}")

fig, axes = plt.subplots(len(states), 2, figsize=(16, 5*len(states)))

if len(states) == 1:
    axes = axes.reshape(1, -1)

for idx, state in enumerate(states):
    if state not in markov_results or 'regime_probs' not in markov_results[state]:
        continue
    
    result = markov_results[state]
    
    # Plot 1: Regime probabilities over time
    ax = axes[idx, 0]
    
    regime_probs = result['regime_probs']
    dates = train_data['date']
    
    ax.plot(dates, regime_probs[0], label='Regime 0', linewidth=2, color='blue', alpha=0.7)
    ax.plot(dates, regime_probs[1], label='Regime 1', linewidth=2, color='red', alpha=0.7)
    ax.fill_between(dates, 0, regime_probs[0], alpha=0.3, color='blue')
    ax.fill_between(dates, 0, regime_probs[1], alpha=0.3, color='red')
    
    ax.set_title(f'{state}: Regime Probabilities Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    
    # Plot 2: Forecast comparison
    ax = axes[idx, 1]
    
    test_dates = result['test_dates']
    actual = result['actual']
    forecast_ms = result['forecast_ms']
    forecast_arimax = result['forecast_arimax']
    
    ax.plot(test_dates, actual, label='Actual', linewidth=2, color='black', marker='o')
    ax.plot(test_dates, forecast_ms, label='Markov Switching', linewidth=2, color='blue', linestyle='--', marker='s')
    ax.plot(test_dates, forecast_arimax, label='Standard ARIMAX', linewidth=2, color='red', linestyle='--', marker='^')
    
    ax.set_title(f'{state}: Forecast Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Claims')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add performance metrics
    ms_r2 = result['metrics_ms']['R2']
    arimax_r2 = result['metrics_arimax']['R2']
    
    textstr = f"Markov R²: {ms_r2:.3f}\nARIMAX R²: {arimax_r2:.3f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('Markov_Switching_Analysis.png', dpi=300, bbox_inches='tight')
print(f" Saved: Markov_Switching_Analysis.png")

print(f"\n{'='*80}")
print("MARKOV SWITCHING ANALYSIS COMPLETE")
print(f"{'='*80}")


