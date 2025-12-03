import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv('FinalFloridaCombinedData.csv')

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# ----------------------------------------------------------
# 2. Create weekly new COVID cases & new deaths
#    (Cumulative → new weekly counts)
# ----------------------------------------------------------
df['New_Confirmed'] = df['Confirmed'].diff().clip(lower=0)
df['New_Deaths'] = df['Deaths'].diff().clip(lower=0)

# Remove the first row that contains NaN for diffs
df_model = df.dropna().copy()

# Target variable (weekly unemployment claims)
y = df_model['claims']

# Exogenous regressors
X = df_model[['New_Confirmed', 'New_Deaths']]

# ----------------------------------------------------------
# 3. Fit forecasting model
#    SARIMAX = ARIMA + exogenous variables
# ----------------------------------------------------------
model = sm.tsa.SARIMAX(
    y,
    exog=X,
    order=(1,1,1),   # simple ARIMA(1,1,1) structure
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

# Show the model summary
print(results.summary())

# ----------------------------------------------------------
# 4. Forecasting next 4 weeks
#
# ----------------------------------------------------------
# Example future values (replace with actual future estimates!)
future_X = pd.DataFrame({
    'New_Confirmed': [1000, 1200, 900, 800],
    'New_Deaths': [50, 55, 40, 35]
})

forecast = results.predict(
    start=len(y),
    end=len(y) + 3,
    exog=future_X
)

print("\n4-week forecast of unemployment claims:")
print(forecast)


def save_text_as_png(text, filename):
    plt.figure(figsize=(8, 6))
    plt.text(0.01, 0.99, text, fontsize=10, va='top', family='monospace')
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

summary_text = results.summary().as_text()
save_text_as_png(summary_text, "sarimax_summary.png")

forecast_df = pd.DataFrame({"forecast": forecast})
save_text_as_png(forecast_df.to_string(), "forecast_output.png")
