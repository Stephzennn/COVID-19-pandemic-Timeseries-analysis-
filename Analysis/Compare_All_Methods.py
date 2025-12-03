import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Compare All ARIMAX Methods
Purpose: Create comprehensive comparison of all approaches
Generates: 
  - Comparison tables
  - Visualization plots
  - Statistical tests
"""

# Create output directory
os.makedirs('Results/comparison', exist_ok=True)

print("="*80)
print("COMPREHENSIVE METHOD COMPARISON")
print("="*80)

# Load all results
baseline = pd.read_csv('Results/baseline/arimax_baseline_results.csv')
outlier_dummies = pd.read_csv('Results/outlier_detection/outlier_dummies_results.csv')
log_transform = pd.read_csv('Results/outlier_detection/log_transform_results.csv')
intervention = pd.read_csv('Results/intervention/arimax_intervention_results.csv')

# Add method identifier
baseline['Method'] = 'Baseline'
outlier_dummies['Method'] = 'Outlier Dummies'
log_transform['Method'] = 'Log Transform'
intervention['Method'] = 'Intervention'

# Combine all results
all_results = pd.concat([baseline, outlier_dummies, log_transform, intervention])

# ============================================
# CREATE COMPARISON TABLE
# ============================================

print("\n" + "="*80)
print("PERFORMANCE COMPARISON - ALL METHODS")
print("="*80)

comparison_table = all_results.pivot_table(
    index='State',
    columns='Method',
    values=['MAE', 'RMSE', 'R2'],
    aggfunc='mean'
)

print("\n" + comparison_table.to_string())

# Save comparison
comparison_table.to_csv('Results/comparison/method_comparison_table.csv')

# ============================================
# VISUALIZE COMPARISON
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: MAE comparison
ax = axes[0, 0]
mae_pivot = all_results.pivot(index='State', columns='Method', values='MAE')
mae_pivot.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_ylabel('MAE')
ax.legend(title='Method', bbox_to_anchor=(1.05, 1))
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: R² comparison
ax = axes[0, 1]
r2_pivot = all_results.pivot(index='State', columns='Method', values='R2')
r2_pivot.plot(kind='bar', ax=ax, width=0.8)
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Baseline (R²=0)')
ax.set_title('R² Comparison (Higher is Better)', fontsize=14, fontweight='bold')
ax.set_ylabel('R²')
ax.legend(title='Method', bbox_to_anchor=(1.05, 1))
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: RMSE comparison
ax = axes[1, 0]
rmse_pivot = all_results.pivot(index='State', columns='Method', values='RMSE')
rmse_pivot.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_ylabel('RMSE')
ax.legend(title='Method', bbox_to_anchor=(1.05, 1))
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Heatmap
ax = axes[1, 1]
heatmap_data = all_results.pivot_table(
    index='State',
    columns='Method',
    values='R2',
    aggfunc='mean'
)
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
            center=0, ax=ax, cbar_kws={'label': 'R²'})
ax.set_title('R² Heatmap by State and Method', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('Results/comparison/method_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================

from scipy import stats

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

# Pairwise t-tests between methods
methods = all_results['Method'].unique()

for i, method1 in enumerate(methods):
    for method2 in methods[i+1:]:
        mae1 = all_results[all_results['Method'] == method1]['MAE']
        mae2 = all_results[all_results['Method'] == method2]['MAE']
        
        if len(mae1) > 1 and len(mae2) > 1:
            t_stat, p_value = stats.ttest_ind(mae1, mae2)
            
            print(f"\n{method1} vs {method2}:")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.3f}")
            
            if p_value < 0.05:
                print(f"   Significant difference (p < 0.05)")
            else:
                print(f"   No significant difference")

# ============================================
# BEST METHOD RECOMMENDATION
# ============================================

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Find best method per state
best_by_state = all_results.loc[all_results.groupby('State')['R2'].idxmax()]

print("\nBest method by state (based on R²):")
print(best_by_state[['State', 'Method', 'MAE', 'RMSE', 'R2']].to_string(index=False))

# Overall best method
overall_best = all_results.groupby('Method')[['MAE', 'R2']].mean()
overall_best = overall_best.sort_values('R2', ascending=False)

print("\nOverall method ranking (average across states):")
print(overall_best)

# Save recommendations
with open('Results/comparison/recommendations.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ARIMAX METHOD RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Best Method Per State:\n")
    f.write(best_by_state[['State', 'Method', 'R2']].to_string(index=False))
    f.write("\n\n")
    
    f.write("Overall Rankings:\n")
    f.write(overall_best.to_string())
    f.write("\n\n")
    
    f.write("Interpretation:\n")
    f.write("- If Baseline is best: Simple model works, no need for complexity\n")
    f.write("- If Outlier Dummies best: Early pandemic spike distorts model\n")
    f.write("- If Log Transform best: Variance changes over time (heteroskedasticity)\n")
    f.write("- If Intervention best: Policy changes have distinct impacts\n")

print("\n Comprehensive comparison complete")
print(" Saved to Results/comparison/")