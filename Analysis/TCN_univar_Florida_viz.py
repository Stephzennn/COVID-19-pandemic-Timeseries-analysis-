import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ahmed's model
from TCN_v1 import fit_tcn, forecast

# Config
HISTORY = 28      # how many past days the model sees
HORIZON = 7       # how many days ahead we forecast

# read input data
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "FinalFloridaCombinedData.csv")
print("Reading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("CSV columns:", df.columns.tolist())
for date_col in ["date", "Date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        break

CLAIMS_COL = "claims"
CASES_COL  = "Confirmed"
DEATHS_COL = "Deaths"

claims = df[CLAIMS_COL].astype(np.float32).values
cases  = df[CASES_COL].astype(np.float32).values
deaths = df[DEATHS_COL].astype(np.float32).values

# Train two separate TCNs: one for claims, one for confirmed cases
claims_train = claims[:-HORIZON]
cases_train  = cases[:-HORIZON]
deaths_train = deaths[:-HORIZON]

print("\nTraining TCN for unemployment claims")
model_claims, hist_claims = fit_tcn(
    claims_train,
    history_length=HISTORY,
    horizon=HORIZON,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
)

print("\nTraining TCN for confirmed COVID-19 cases")
model_cases, hist_cases = fit_tcn(
    cases_train,
    history_length=HISTORY,
    horizon=HORIZON,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
)

print("\nTraining TCN for COVID-19 deaths")
model_deaths, hist_deaths = fit_tcn(
    deaths_train,
    history_length=HISTORY,
    horizon=HORIZON,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
)

# Build context windows to compare forecast vs actual on those held-out days
ctx_claims = claims[-(HISTORY + HORIZON) : -HORIZON]
ctx_cases  = cases[-(HISTORY + HORIZON) : -HORIZON]
ctx_deaths = deaths[-(HISTORY + HORIZON) : -HORIZON]

pred_claims = forecast(model_claims, ctx_claims)
pred_cases  = forecast(model_cases, ctx_cases)
pred_deaths = forecast(model_deaths, ctx_deaths)

# Ground truth for those last HORIZON days
future_dates = df.index[-HORIZON:]
true_claims  = claims[-HORIZON:]
true_cases   = cases[-HORIZON:]
true_deaths  = deaths[-HORIZON:]

# Plot: Claims
plt.figure(figsize=(10, 4))
plt.plot(future_dates, true_claims, label="Actual claims")
plt.plot(future_dates, pred_claims, "--", label="TCN forecast")
plt.title("Florida – Initial Unemployment Claims (TCN forecast)")
plt.xlabel("Date")
plt.ylabel("Claims")
plt.legend()
plt.tight_layout()

# Plot: Confirmed cases
plt.figure(figsize=(10, 4))
plt.plot(future_dates, true_cases, label="Actual confirmed cases")
plt.plot(future_dates, pred_cases, "--", label="TCN forecast")
plt.title("Florida – Confirmed COVID-19 Cases (TCN forecast)")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()

# Plot: deaths
plt.figure(figsize=(10, 4))
plt.plot(future_dates, true_deaths, label="Actual deaths")
plt.plot(future_dates, pred_deaths, "--", label="TCN forecast")
plt.title("Florida – COVID-19 Deaths (TCN forecast)")
plt.xlabel("Date")
plt.ylabel("Deaths")
plt.legend()
plt.tight_layout()

plt.show()
