import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#Import data
unemployment = pd.read_csv('data/state_unemployment_claims_long.csv')
df = unemployment.copy()
df['Date'] = pd.to_datetime(df['Date'])

#Plotting
fig = plt.figure(figsize=(16,12))

#1. National trend of unemployment claims over time
ax1 = fig.add_subplot(3,1,1)
national = df.groupby('Date')['Initial_Claims'].sum().reset_index()
ax1.plot(national['Date'], national['Initial_Claims'], linewidth=2, color='blue')
ax1.fill_between(national['Date'], national['Initial_Claims'], color='blue', alpha=0.1)
ax1.set_title('National Unemployment Claims Over Time', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Claims')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

#2. State-wise unemployment claims over time
ax2 = fig.add_subplot(3,2,2)
top_states = df.groupby('Province_State')['Initial_Claims'].mean().nlargest(10).sort_values()
top_states.plot(kind='barh', ax=ax2, color='orange')
ax2.set_title('Top 10 States by Average Unemployment Claims', fontsize=12)
ax2.set_xlabel('Average Number of Claims')
ax2.set_ylabel('State')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', rotation=0)

#3. Monthly trend of unemployment claims by state
ax3 = fig.add_subplot(3,2,3)
major_states = ['CA', 'NY', 'TX', 'GA', 'FL']
for state in major_states:
    state_data = df[df['Province_State'] == state].groupby('Date')['Initial_Claims'].sum().reset_index()
    ax3.plot(state_data['Date'], state_data['Initial_Claims'], 
             label=state, linewidth=1.5, alpha=0.7, marker='o', markersize=2)
ax3.set_title('Unemployment Trend by Major States', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Claims')
ax3.legend(title='State')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

#4. Month boxplot of unemployment claims (to show seasonality/volatility)
ax4 = fig.add_subplot(3,2,4)
df['Month'] = df['Date'].dt.month
sns.boxplot(x='Month', y='Initial_Claims', data=df, ax=ax4, palette='Set2')
ax4.set_title('Monthly Distribution of Unemployment Claims', fontsize=12)
ax4.set_xlabel('Month')
ax4.set_ylabel('Claims')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1,13))
ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

#5. Year over year comparison
ax5 = fig.add_subplot(3,2,5)
df['Year'] = df['Date'].dt.year

for year in sorted(df['Year'].unique()):
    year_data = df[df['Year'] == year].sort_values('Date')
    year_data['Week of Year'] = year_data['Date'].dt.isocalendar().week
    ax5.plot(year_data['Week of Year'], year_data['Initial_Claims'].groupby(year_data['Week of Year']).transform('mean'),
             label=year, linewidth=1.5, alpha=0.7, marker='o', markersize=2)
ax5.set_title('Year over Year Comparison', fontsize=12)
ax5.set_xlabel('Week of Year')
ax5.set_ylabel('Average Claims')
ax5.legend()
ax5.grid(True, alpha=0.3)

#6. Heatmap of unemployment claims by state and month
ax6 = plt.subplot(3, 2, 6)
peak_by_state = df.groupby('Province_State')['Initial_Claims'].max().sort_values(ascending=False)
norm_values = peak_by_state / peak_by_state.max()
cmap = plt.cm.get_cmap('viridis')
colors = [cmap(val) for val in norm_values]
ax6.barh(peak_by_state.index, peak_by_state.values, color=colors)
ax6.set_title('Peak Initial Claims by State (Entire Period)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Peak Claims')

plt.tight_layout()
plt.show()
plt.savefig('unemployment_analysis.png')
print("Figure saved as 'unemployment_analysis.png'")