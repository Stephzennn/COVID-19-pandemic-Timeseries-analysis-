import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ahmed's model
from TCN_v1 import fit_tcn, forecast

HISTORY = 28
TRAIN_START_DATE = pd.Timestamp("2021-09-01")
VAL_START_DATE   = pd.Timestamp("2022-04-01")
TEST_START_DATE  = pd.Timestamp("2022-10-01")

# Read data
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

dates_index = df.index             
dates = dates_index.to_numpy()     

n = len(claims)
if n <= HISTORY + 5:  
    raise ValueError("Series too short for chosen HISTORY.")

train_start_idx = int(dates_index.searchsorted(TRAIN_START_DATE, side="left"))
val_start_idx   = int(dates_index.searchsorted(VAL_START_DATE,   side="left"))
test_start_idx  = int(dates_index.searchsorted(TEST_START_DATE,  side="left"))
test_end_idx    = n - 1

train_end_idx = val_start_idx - 1
val_end_idx   = test_start_idx - 1

if train_start_idx >= val_start_idx:
    raise ValueError("TRAIN_START_DATE must be before VAL_START_DATE.")
if val_start_idx >= test_start_idx:
    raise ValueError("VAL_START_DATE must be before TEST_START_DATE.")
if test_start_idx >= n:
    raise ValueError("TEST_START_DATE is after the data ends.")

HORIZON = test_end_idx - test_start_idx + 1

print(f"Train indices: {train_start_idx}..{train_end_idx} "
      f"({dates[train_start_idx]}..{dates[train_end_idx]})")
print(f"Val indices:   {val_start_idx}..{val_end_idx} "
      f"({dates[val_start_idx]}..{dates[val_end_idx]})")
print(f"Test indices:  {test_start_idx}..{test_end_idx} "
      f"({dates[test_start_idx]}..{dates[test_end_idx]})")
print(f"HORIZON (test length) = {HORIZON}")

TRAIN_TAIL_LEN = 60
train_tail_start_idx = max(train_start_idx, train_end_idx - TRAIN_TAIL_LEN + 1)

claims_train_all = claims[train_start_idx:test_start_idx]
cases_train_all  = cases[train_start_idx:test_start_idx]
deaths_train_all = deaths[train_start_idx:test_start_idx]

print("\nTraining TCN for unemployment claims")
model_claims, hist_claims = fit_tcn(
    claims_train_all,
    history_length=HISTORY,
    horizon=HORIZON,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
    val_ratio=0.0,        
)

print("\nTraining TCN for confirmed COVID-19 cases")
model_cases, hist_cases = fit_tcn(
    cases_train_all,
    history_length=HISTORY,
    horizon=HORIZON,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
    val_ratio=0.0,        
)

print("\nTraining TCN for COVID-19 deaths")
model_deaths, hist_deaths = fit_tcn(
    deaths_train_all,
    history_length=HISTORY,
    horizon=HORIZON,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
    val_ratio=0.0,      
)

ctx_claims = claims_train_all[-HISTORY:]
ctx_cases  = cases_train_all[-HISTORY:]
ctx_deaths = deaths_train_all[-HISTORY:]

pred_claims = forecast(model_claims, ctx_claims) 
pred_cases  = forecast(model_cases, ctx_cases)
pred_deaths = forecast(model_deaths, ctx_deaths)

test_dates  = dates[test_start_idx:test_end_idx + 1]
true_claims = claims[test_start_idx:test_end_idx + 1]
true_cases  = cases[test_start_idx:test_end_idx + 1]
true_deaths = deaths[test_start_idx:test_end_idx + 1]

# Helper: single panel with AF-style coloring and vertical lines
def plot_series_panel(title, y, y_pred):
    plt.figure(figsize=(10, 4))

    # Train (tail)
    plt.plot(
        dates[train_tail_start_idx:train_end_idx + 1],
        y[train_tail_start_idx:train_end_idx + 1],
        label="Train (Tail)",
        color="C0",
    )

    # Validation
    plt.plot(
        dates[val_start_idx:val_end_idx + 1],
        y[val_start_idx:val_end_idx + 1],
        label="Validation",
        color="C1",
    )

    # Test actual
    plt.plot(
        dates[test_start_idx:test_end_idx + 1],
        y[test_start_idx:test_end_idx + 1],
        label="Test Actual",
        color="C2",
    )

    # Forecast on test window
    plt.plot(
        dates[test_start_idx:test_end_idx + 1],
        y_pred,
        label="Forecast (Test)",
        color="C3",
    )

    # Vertical lines at train/val and val/test boundaries
    plt.axvline(dates[val_start_idx], color="k", linestyle="--", linewidth=1)
    plt.axvline(dates[test_start_idx], color="k", linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(title.split("–")[-1].strip())
    plt.legend()
    plt.tight_layout()

# Plots for each series
plot_series_panel(
    "CLAIMS – Train Tail + Validation + Test Forecast (TCN)",
    claims,
    pred_claims,
)

plot_series_panel(
    "CONFIRMED CASES – Train Tail + Validation + Test Forecast (TCN)",
    cases,
    pred_cases,
)

plot_series_panel(
    "DEATHS – Train Tail + Validation + Test Forecast (TCN)",
    deaths,
    pred_deaths,
)

plt.show()
