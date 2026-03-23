"""
NSE Stock Price Predictor v2
Key fix: Predict LOG RETURNS (not absolute prices) + Multi-task Learning
Author: Rebuilt for accuracy
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, GRU, Dense, Dropout,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Add, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBClassifier, XGBRegressor

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
TICKER         = "HDFCBANK.NS"     # ← change ticker here only
INTERVAL       = "1h"
PERIOD         = "730d"
LOOKBACK       = 24            # 1 trading day lookback (reduced: less noise)
FORECAST_HOURS = 18
TRAIN_SPLIT    = 0.80
SEED           = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════
print("=" * 60)
print(f"  {TICKER} Hourly Prediction Pipeline  v2")
print("=" * 60)
print("\n[1/7] Downloading data...")

raw = yf.download(TICKER, interval=INTERVAL, period=PERIOD, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()
df.index = df.index.tz_convert("Asia/Kolkata")
print(f"    Rows downloaded: {len(df)}")


# ══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("\n[2/7] Engineering features...")

close  = df["Close"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
volume = df["Volume"].squeeze()
open_  = df["Open"].squeeze()

# ── Core returns (THE key signal — not raw price)
df["log_return"]      = np.log(close / close.shift(1))
df["open_log_return"] = np.log(open_ / close.shift(1))   # overnight gap
df["HL_pct"]          = (high - low) / close             # candle range %
df["OC_pct"]          = (close - open_) / open_          # body direction

# ── Lagged returns (most predictive for next-step)
for lag in [1, 2, 3, 6, 12, 24]:
    df[f"ret_lag{lag}"] = df["log_return"].shift(lag)

# ── Rolling return stats
df["ret_mean_6"]  = df["log_return"].rolling(6).mean()
df["ret_mean_12"] = df["log_return"].rolling(12).mean()
df["ret_std_6"]   = df["log_return"].rolling(6).std()
df["ret_std_12"]  = df["log_return"].rolling(12).std()

# ── Momentum
df["RSI_14"]  = ta.momentum.rsi(close, window=14)
df["RSI_6"]   = ta.momentum.rsi(close, window=6)
stoch         = ta.momentum.StochasticOscillator(high, low, close)
df["Stoch_K"] = stoch.stoch()
df["Stoch_D"] = stoch.stoch_signal()
df["ROC_5"]   = ta.momentum.roc(close, window=5)
df["ROC_10"]  = ta.momentum.roc(close, window=10)

# ── Trend (normalised — ratio to price, not absolute)
df["SMA_10_ratio"] = close / ta.trend.sma_indicator(close, window=10)
df["SMA_20_ratio"] = close / ta.trend.sma_indicator(close, window=20)
df["EMA_10_ratio"] = close / ta.trend.ema_indicator(close, window=10)

macd = ta.trend.MACD(close)
df["MACD_diff"]  = macd.macd_diff()           # histogram only (normalised internally)
df["MACD_cross"] = (macd.macd() > macd.macd_signal()).astype(int)

# ── Volatility
bb           = ta.volatility.BollingerBands(close)
df["BB_pct"] = bb.bollinger_pband()           # 0–1 position inside bands
df["BB_width"]= (bb.bollinger_hband() - bb.bollinger_lband()) / close
df["ATR_pct"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range() / close

# ── Volume
df["vol_ratio"]   = volume / volume.rolling(20).mean()
df["vol_log"]     = np.log1p(volume)
df["OBV_norm"]    = ta.volume.on_balance_volume(close, volume) / 1e6

# ── Time (cyclical — hour of day drives open/close gap patterns)
df["hour_sin"]    = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df.index.hour / 24)
df["dow_sin"]     = np.sin(2 * np.pi * df.index.dayofweek / 5)
df["dow_cos"]     = np.cos(2 * np.pi * df.index.dayofweek / 5)
df["is_first_hour"] = (df.index.hour == 9).astype(int)   # 9:15 open surge
df["is_last_hour"]  = (df.index.hour == 15).astype(int)  # 15:00 close rush

# ══════════════════════════════════════════════════════════
# TARGETS — predict RETURNS, reconstruct prices later
# ══════════════════════════════════════════════════════════
# Next candle's open and close as % change from current close
df["tgt_open_ret"]  = np.log(df["Open"].shift(-1)  / close)   # log(next_open  / curr_close)
df["tgt_close_ret"] = np.log(df["Close"].shift(-1) / close)   # log(next_close / curr_close)

# Direction labels (1 = up, 0 = down/flat) — for classifier
df["tgt_open_dir"]  = (df["tgt_open_ret"]  > 0).astype(int)
df["tgt_close_dir"] = (df["tgt_close_ret"] > 0).astype(int)

# Keep the actual future prices for evaluation reconstruction
df["future_open"]  = df["Open"].shift(-1)
df["future_close"] = df["Close"].shift(-1)

df.dropna(inplace=True)
print(f"    Rows after engineering: {len(df)}")

FEATURE_COLS   = [c for c in df.columns if c.startswith(("log_", "open_log", "HL_", "OC_",
                                                           "ret_", "RSI", "Stoch", "ROC",
                                                           "SMA_", "EMA_", "MACD", "BB_",
                                                           "ATR", "vol_", "OBV", "hour",
                                                           "dow", "is_"))]
TARGET_RET     = ["tgt_open_ret", "tgt_close_ret"]
TARGET_DIR     = ["tgt_open_dir", "tgt_close_dir"]
PRICE_COLS     = ["future_open", "future_close"]

print(f"    Feature columns: {len(FEATURE_COLS)}")


# ══════════════════════════════════════════════════════════
# 3. SCALE & SEQUENCES
# ══════════════════════════════════════════════════════════
print("\n[3/7] Preparing sequences...")

scaler_X = RobustScaler()
scaler_y = RobustScaler()   # scale returns too (they're already small, helps LSTM)

X_scaled = scaler_X.fit_transform(df[FEATURE_COLS])
y_ret_scaled = scaler_y.fit_transform(df[TARGET_RET])
y_dir  = df[TARGET_DIR].values
y_prices = df[PRICE_COLS].values     # kept raw for final evaluation
curr_close = df["Close"].values      # current close for price reconstruction


def create_sequences(X, y_ret, y_dir, y_prices, curr_close, lookback):
    Xs, yrets, ydirs, yprices, closes = [], [], [], [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        yrets.append(y_ret[i])
        ydirs.append(y_dir[i])
        yprices.append(y_prices[i])
        closes.append(curr_close[i])
    return (np.array(Xs), np.array(yrets), np.array(ydirs),
            np.array(yprices), np.array(closes))


X_seq, y_ret_seq, y_dir_seq, y_price_seq, close_seq = create_sequences(
    X_scaled, y_ret_scaled, y_dir, y_prices, curr_close, LOOKBACK
)

split = int(len(X_seq) * TRAIN_SPLIT)
X_train, X_test     = X_seq[:split],       X_seq[split:]
yret_train, yret_test = y_ret_seq[:split], y_ret_seq[split:]
ydir_train, ydir_test = y_dir_seq[:split], y_dir_seq[split:]
yprice_test           = y_price_seq[split:]
close_test            = close_seq[split:]

print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
n_features = X_train.shape[2]


# ══════════════════════════════════════════════════════════
# 4. MODEL: Multi-Task BiLSTM
#    Head 1 → regression (predict return)
#    Head 2 → binary classification (predict direction)
# ══════════════════════════════════════════════════════════
print("\n[4/7] Building multi-task BiLSTM model...")

def build_multitask_model(lookback, n_features):
    inp = Input(shape=(lookback, n_features), name="sequence_input")

    # Shared encoder
    x = Bidirectional(LSTM(96, return_sequences=True))(inp)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # Attention
    attn = MultiHeadAttention(num_heads=4, key_dim=24)(x, x)
    x    = LayerNormalization()(Add()([x, attn]))
    x    = GlobalAveragePooling1D()(x)

    shared = Dense(64, activation="relu")(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.2)(shared)

    # ── Regression head (returns)
    reg = Dense(32, activation="relu")(shared)
    reg_out = Dense(2, name="return_output")(reg)     # open_ret, close_ret

    # ── Direction head (classification)
    clf = Dense(32, activation="relu")(shared)
    clf_out = Dense(2, activation="sigmoid", name="dir_output")(clf)  # open_dir, close_dir

    model = Model(inp, [reg_out, clf_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={
            "return_output": "huber",
            "dir_output":    "binary_crossentropy",
        },
        loss_weights={
            "return_output": 1.0,
            "dir_output":    0.5,    # secondary objective
        },
        metrics={
            "dir_output": "accuracy"
        }
    )
    return model


model = build_multitask_model(LOOKBACK, n_features)
model.summary()

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5, monitor="val_loss")
]

print("\nTraining...")
history = model.fit(
    X_train,
    {"return_output": yret_train, "dir_output": ydir_train},
    epochs=80,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)


# ══════════════════════════════════════════════════════════
# 5. XGBoost DIRECTION CLASSIFIER (standalone, for comparison)
#    Uses tabular features only — often beats LSTM on direction
# ══════════════════════════════════════════════════════════
print("\n[5/7] Training XGBoost direction classifiers...")

X_tab_train = X_train[:, -1, :]   # last timestep
X_tab_test  = X_test[:, -1, :]

xgb_params = dict(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=5,           # prevents overfitting on small moves
    gamma=0.1,
    random_state=SEED,
    tree_method="hist",
    eval_metric="logloss"
)

xgb_open_dir  = XGBClassifier(**xgb_params)
xgb_close_dir = XGBClassifier(**xgb_params)

xgb_open_dir.fit(
    X_tab_train, ydir_train[:, 0],
    eval_set=[(X_tab_test, ydir_test[:, 0])], verbose=False
)
xgb_close_dir.fit(
    X_tab_train, ydir_train[:, 1],
    eval_set=[(X_tab_test, ydir_test[:, 1])], verbose=False
)


# ══════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════
print("\n[6/7] Evaluating...")

# LSTM predictions
lstm_ret_pred, lstm_dir_pred = model.predict(X_test, verbose=0)
lstm_ret_inv = scaler_y.inverse_transform(lstm_ret_pred)   # back to log-returns

# Reconstruct absolute prices from predicted returns
lstm_pred_open  = close_test * np.exp(lstm_ret_inv[:, 0])
lstm_pred_close = close_test * np.exp(lstm_ret_inv[:, 1])

actual_open  = yprice_test[:, 0]
actual_close = yprice_test[:, 1]

# XGBoost direction predictions
xgb_pred_open_dir  = xgb_open_dir.predict(X_tab_test)
xgb_pred_close_dir = xgb_close_dir.predict(X_tab_test)

# LSTM binary direction (from sigmoid head, threshold 0.5)
lstm_dir_open  = (lstm_dir_pred[:, 0] > 0.5).astype(int)
lstm_dir_close = (lstm_dir_pred[:, 1] > 0.5).astype(int)

# Helper metrics
def rmse(a, b): return np.sqrt(mean_squared_error(a, b))
def mape(a, b): return np.mean(np.abs((a - b) / (np.abs(a) + 1e-9))) * 100

print("\n" + "=" * 65)
print("  PRICE PREDICTION METRICS (Test Set)")
print("=" * 65)
print(f"\n  {'Metric':<30} {'OPEN':>12} {'CLOSE':>12}")
print("  " + "-" * 55)
print(f"  {'RMSE  (₹)':<30} {rmse(actual_open, lstm_pred_open):>12.2f} {rmse(actual_close, lstm_pred_close):>12.2f}")
print(f"  {'MAE   (₹)':<30} {mean_absolute_error(actual_open, lstm_pred_open):>12.2f} {mean_absolute_error(actual_close, lstm_pred_close):>12.2f}")
print(f"  {'MAPE  (%)':<30} {mape(actual_open, lstm_pred_open):>12.2f} {mape(actual_close, lstm_pred_close):>12.2f}")

print("\n" + "=" * 65)
print("  DIRECTION ACCURACY (Test Set)  — random baseline = 50%")
print("=" * 65)
print(f"\n  {'Model':<30} {'OPEN DIR':>12} {'CLOSE DIR':>12}")
print("  " + "-" * 55)
print(f"  {'BiLSTM (multi-task)':<30} {accuracy_score(ydir_test[:,0], lstm_dir_open)*100:>11.2f}% {accuracy_score(ydir_test[:,1], lstm_dir_close)*100:>11.2f}%")
print(f"  {'XGBoost standalone':<30} {accuracy_score(ydir_test[:,0], xgb_pred_open_dir)*100:>11.2f}% {accuracy_score(ydir_test[:,1], xgb_pred_close_dir)*100:>11.2f}%")

# Ensemble direction: agree → use that, disagree → trust XGBoost
ensemble_open_dir  = np.where(lstm_dir_open  == xgb_pred_open_dir,  lstm_dir_open,  xgb_pred_open_dir)
ensemble_close_dir = np.where(lstm_dir_close == xgb_pred_close_dir, lstm_dir_close, xgb_pred_close_dir)
print(f"  {'Ensemble (agree=confident)':<30} {accuracy_score(ydir_test[:,0], ensemble_open_dir)*100:>11.2f}% {accuracy_score(ydir_test[:,1], ensemble_close_dir)*100:>11.2f}%")
print("=" * 65)

print("\n  Close Direction Detail:")
print(classification_report(ydir_test[:, 1], ensemble_close_dir, target_names=["Down", "Up"]))


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════
sns.set_style("darkgrid")

# Plot 1: Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history["loss"],     label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["dir_output_accuracy"],     label="Train Dir Acc")
axes[1].plot(history.history["val_dir_output_accuracy"], label="Val Dir Acc")
axes[1].axhline(0.5, color="gray", linestyle="--", label="Random baseline")
axes[1].set_title("Direction Accuracy During Training")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle(f"{TICKER} — Training Curves", fontweight="bold")
plt.tight_layout()
##plt.savefig("/mnt/user-data/outputs/v2_training_curves.png", dpi=150)
plt.show()

# Plot 2: Predicted vs Actual (last 150 test candles)
N = min(150, len(actual_close))
test_idx = df.index[LOOKBACK + split: LOOKBACK + split + N]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, actual, pred, title in [
    (axes[0], actual_open[:N],  lstm_pred_open[:N],  "Open"),
    (axes[1], actual_close[:N], lstm_pred_close[:N], "Close"),
]:
    ax.plot(test_idx, actual, label="Actual",    color="steelblue",  lw=1.5)
    ax.plot(test_idx, pred,   label="Predicted", color="tomato",     lw=1.5, linestyle="--")
    err = np.abs(actual - pred) / actual * 100
    ax.fill_between(test_idx, actual, pred, alpha=0.15, color="orange")
    ax.set_title(f"{TICKER} — {title} Price  |  Avg Error: {err.mean():.2f}%")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
##plt.savefig("/mnt/user-data/outputs/v2_predicted_vs_actual.png", dpi=150)
plt.show()

# Plot 3: XGBoost feature importance (top 20)
feat_imp = pd.Series(xgb_close_dir.feature_importances_, index=FEATURE_COLS)
feat_imp = feat_imp.nlargest(20)

plt.figure(figsize=(10, 6))
feat_imp.sort_values().plot(kind="barh", color="steelblue", edgecolor="white")
plt.title(f"{TICKER} — XGBoost Close Direction: Top 20 Features")
plt.xlabel("Feature Importance")
plt.tight_layout()
##plt.savefig("/mnt/user-data/outputs/v2_feature_importance.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════
# 7. FORECAST: Next ~3 trading days
# ══════════════════════════════════════════════════════════
print(f"\n[7/7] Generating {FORECAST_HOURS}-hour forecast...")

current_seq = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, n_features)
current_tab = X_scaled[-1:].copy()
last_close  = df["Close"].iloc[-1]
last_time   = df.index[-1]
records     = []

while len(records) < FORECAST_HOURS:
    last_time += pd.Timedelta(hours=1)

    if last_time.weekday() >= 5:
        continue
    if last_time.hour < 9 or (last_time.hour == 9 and last_time.minute < 15):
        continue
    if last_time.hour > 15 or (last_time.hour == 15 and last_time.minute > 30):
        continue

    # Predict returns
    ret_pred_s, dir_pred = model.predict(current_seq, verbose=0)
    ret_pred = scaler_y.inverse_transform(ret_pred_s)[0]

    pred_open  = last_close * np.exp(ret_pred[0])
    pred_close = last_close * np.exp(ret_pred[1])

    # XGBoost direction vote
    xgb_open_d  = xgb_open_dir.predict(current_tab)[0]
    xgb_close_d = xgb_close_dir.predict(current_tab)[0]

    # Confidence from both models
    lstm_open_conf  = float(dir_pred[0, 0])
    lstm_close_conf = float(dir_pred[0, 1])

    # Ensemble: if both models agree, confidence is high
    open_dir_agree  = int(xgb_open_d) == int(lstm_open_conf > 0.5)
    close_dir_agree = int(xgb_close_d) == int(lstm_close_conf > 0.5)

    records.append({
        "Datetime":         last_time,
        "Pred_Open":        round(pred_open, 2),
        "Pred_Close":       round(pred_close, 2),
        "Open_Direction":   "▲ UP" if xgb_open_d == 1  else "▼ DOWN",
        "Close_Direction":  "▲ UP" if xgb_close_d == 1 else "▼ DOWN",
        "High_Confidence":  open_dir_agree and close_dir_agree,
    })

    # Update last_close for chaining
    last_close = pred_close

    # Slide sequence
    next_feat = df[FEATURE_COLS].iloc[-1].copy()
    next_feat["log_return"]      = ret_pred[1]
    next_feat["open_log_return"] = ret_pred[0]
    next_feat["hour_sin"]        = np.sin(2 * np.pi * last_time.hour / 24)
    next_feat["hour_cos"]        = np.cos(2 * np.pi * last_time.hour / 24)
    next_feat["is_first_hour"]   = int(last_time.hour == 9)
    next_feat["is_last_hour"]    = int(last_time.hour == 15)

    next_scaled = scaler_X.transform(next_feat.values.reshape(1, -1))
    current_tab = next_scaled
    current_seq = np.append(current_seq[:, 1:, :], next_scaled.reshape(1, 1, -1), axis=1)


forecast_df = pd.DataFrame(records).set_index("Datetime")

# Daily summary
daily_df = forecast_df.groupby(forecast_df.index.date).agg(
    Open=("Pred_Open",       "first"),
    Close=("Pred_Close",     "last"),
    Change_pct=("Pred_Close", lambda x:
        round((x.iloc[-1] / forecast_df.loc[forecast_df.index.date == x.index[0].date(), "Pred_Open"].iloc[0] - 1) * 100, 2))
)

print("\n" + "=" * 60)
print(f"  FORECAST — {TICKER}  (Hourly)")
print("=" * 60)
print(forecast_df[["Pred_Open", "Pred_Close", "Open_Direction", "Close_Direction", "High_Confidence"]].to_string())

print("\n" + "=" * 60)
print(f"  FORECAST — {TICKER}  (Daily Summary)")
print("=" * 60)
print(daily_df.to_string())

# Forecast plot
fig, ax = plt.subplots(figsize=(13, 5))
context = df["Close"].iloc[-LOOKBACK * 2:]
ax.plot(context.index, context.values, color="steelblue", lw=1.8, label="Actual (context)")

colors = forecast_df["High_Confidence"].map({True: "tomato", False: "salmon"})
ax.scatter(forecast_df.index, forecast_df["Pred_Open"],  marker="o", color="orange", s=40, zorder=3, label="Pred Open")
ax.scatter(forecast_df.index, forecast_df["Pred_Close"], marker="s", color="tomato",  s=40, zorder=3, label="Pred Close")
ax.plot(forecast_df.index, forecast_df["Pred_Close"], color="tomato", lw=1.2, linestyle="--", alpha=0.7)

ax.axvline(x=df.index[-1], color="gray", linestyle=":", lw=1.5, label="Forecast start")
ax.set_title(f"{TICKER} — {FORECAST_HOURS}h Forecast  |  ● High conf  ○ Low conf", fontweight="bold")
ax.set_ylabel("Price (₹)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
##plt.savefig("/mnt/user-data/outputs/v2_forecast.png", dpi=150)
plt.show()

print("\n✓ All outputs saved to /mnt/user-data/outputs/")
# NEXT CELL #
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════
# INTERACTIVE FORECAST PLOT (PLOTLY)
# ══════════════════════════════════════════════════════════

# 1. Grab historical context
context = df["Close"].iloc[-LOOKBACK * 2:]

fig = go.Figure()

# 2. Plot Historical Actuals
fig.add_trace(go.Scatter(
    x=context.index,
    y=context.values,
    mode='lines',
    name='Actual Close (Context)',
    line=dict(color='#3498db', width=2),
    hovertemplate="Actual: ₹%{y:.2f}<extra></extra>"
))

# 3. Plot Predicted Open
fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["Pred_Open"],
    mode='markers',
    name='Predicted Open',
    marker=dict(
        color='#f39c12', 
        size=8, 
        symbol='circle',
        line=dict(width=1, color='white')
    ),
    hovertemplate="Pred Open: ₹%{y:.2f}<extra></extra>"
))

# 4. Plot Predicted Close (with connecting line)
# We map the confidence boolean to a text array for the hover tooltip
conf_text = forecast_df["High_Confidence"].map({True: "High", False: "Low"})

fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["Pred_Close"],
    mode='lines+markers',
    name='Predicted Close',
    line=dict(color='#e74c3c', width=2, dash='dash'),
    marker=dict(
        color='#e74c3c', 
        size=8, 
        symbol='square',
        line=dict(width=1, color='white')
    ),
    text=conf_text,
    hovertemplate="Pred Close: ₹%{y:.2f}<br>Model Confidence: %{text}<extra></extra>"
))

# 5. Add a vertical line to mark where the forecast begins
fig.add_vline(
    x=df.index[-1].timestamp() * 1000, # Plotly uses ms timestamps for vlines
    line_width=2, 
    line_dash="dot", 
    line_color="#95a5a6",
    annotation_text="Forecast Start", 
    annotation_position="top left"
)

# 6. Apply a modern, professional layout
fig.update_layout(
    title=dict(
        text=f"<b>{TICKER} — {FORECAST_HOURS}h Interactive Forecast</b>",
        font=dict(size=20)
    ),
    xaxis_title="Datetime",
    yaxis_title="Price (₹)",
    template="plotly_dark",  # Sleek, minimalist dark mode
    hovermode="x unified",   # Shows all data for a specific hour in one tooltip
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=40, r=40, t=80, b=40)
)

# Render the interactive chart
fig.show()
# NEXT CELL #
"""
NSE Stock Price Predictor v4
Fixes vs v3:
  1. Return scaling   — targets multiplied ×100 (percent space, not raw log-return)
                        prevents Huber-loss lazy-zero collapse
  2. Direction head   — separated into two independent binary outputs with
                        individual loss terms; fixes the ~33% training accuracy
  3. Forecast chain   — when |predicted return| < ATR noise floor, replace with
                        ATR-scaled directional step so price never flatlines
  4. Return clipping  — predictions clipped to ±3σ of training return distribution
  5. Huber delta      — reduced to 0.5 (was 1.0 default) to penalise near-zero
                        predictions harder in percent space
  6. Loss weights     — regression weight raised to 2.0, direction to 1.0 each
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              accuracy_score, classification_report)
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Add, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
TICKER         = "HDFCBANK.NS"
INTERVAL       = "1h"
PERIOD         = "730d"
LOOKBACK       = 24
FORECAST_HOURS = 18
TRAIN_SPLIT    = 0.80
SEED           = 42

# FIX 1: predict in PERCENT space (×100) so Huber loss has signal
RETURN_SCALE   = 100.0

tf.random.set_seed(SEED)
np.random.seed(SEED)


def is_nse_open(ts: pd.Timestamp) -> bool:
    if ts.weekday() >= 5:
        return False
    minutes = ts.hour * 60 + ts.minute
    return 555 <= minutes <= 930   # 09:15–15:30 IST


# ══════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════
print("=" * 60)
print(f"  {TICKER} Hourly Prediction Pipeline  v4")
print("=" * 60)
print("\n[1/8] Downloading data...")

raw = yf.download(TICKER, interval=INTERVAL, period=PERIOD, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
rows_raw = len(df)
df.dropna(inplace=True)
print(f"    Rows downloaded  : {rows_raw}")
print(f"    Dropped (NaN)    : {rows_raw - len(df)}")

df.index = df.index.tz_convert("Asia/Kolkata")
rows_before_mkt = len(df)
df = df[df.index.map(is_nse_open)]
print(f"    Dropped (non-mkt): {rows_before_mkt - len(df)}")
print(f"    Rows remaining   : {len(df)}")


# ══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("\n[2/8] Engineering features...")

close  = df["Close"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
volume = df["Volume"].squeeze()
open_  = df["Open"].squeeze()

df["log_return"]      = np.log(close  / close.shift(1))
df["open_log_return"] = np.log(open_  / close.shift(1))
df["HL_pct"]          = (high - low)   / close
df["OC_pct"]          = (close - open_) / open_

rows_before_cb = len(df)
df = df[df["log_return"].abs() < 0.15]
print(f"    Dropped (circuit-breaker): {rows_before_cb - len(df)}")

close  = df["Close"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
volume = df["Volume"].squeeze()
open_  = df["Open"].squeeze()

for lag in [1, 2, 3, 6, 12, 24]:
    df[f"ret_lag{lag}"] = df["log_return"].shift(lag)

df["ret_mean_6"]  = df["log_return"].rolling(6).mean()
df["ret_mean_12"] = df["log_return"].rolling(12).mean()
df["ret_std_6"]   = df["log_return"].rolling(6).std()
df["ret_std_12"]  = df["log_return"].rolling(12).std()

df["RSI_14"]  = ta.momentum.rsi(close, window=14)
df["RSI_6"]   = ta.momentum.rsi(close, window=6)
stoch         = ta.momentum.StochasticOscillator(high, low, close)
df["Stoch_K"] = stoch.stoch()
df["Stoch_D"] = stoch.stoch_signal()
df["ROC_5"]   = ta.momentum.roc(close, window=5)
df["ROC_10"]  = ta.momentum.roc(close, window=10)

df["SMA_10_ratio"] = close / ta.trend.sma_indicator(close, window=10)
df["SMA_20_ratio"] = close / ta.trend.sma_indicator(close, window=20)
df["EMA_10_ratio"] = close / ta.trend.ema_indicator(close, window=10)

macd = ta.trend.MACD(close)
df["MACD_diff"]  = macd.macd_diff()
df["MACD_cross"] = (macd.macd() > macd.macd_signal()).astype(int)

bb            = ta.volatility.BollingerBands(close)
df["BB_pct"]  = bb.bollinger_pband()
df["BB_width"]= (bb.bollinger_hband() - bb.bollinger_lband()) / close
df["ATR_pct"] = (ta.volatility.AverageTrueRange(high, low, close)
                 .average_true_range() / close)

df["vol_ratio"] = volume / volume.rolling(20).mean()
df["vol_log"]   = np.log1p(volume)

obv_raw        = ta.volume.on_balance_volume(close, volume)
obv_med        = obv_raw.rolling(50).median().replace(0, np.nan)
df["OBV_norm"] = obv_raw / obv_med

df["hour_sin"]      = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"]      = np.cos(2 * np.pi * df.index.hour / 24)
df["dow_sin"]       = np.sin(2 * np.pi * df.index.dayofweek / 5)
df["dow_cos"]       = np.cos(2 * np.pi * df.index.dayofweek / 5)
df["is_first_hour"] = (df.index.hour == 9).astype(int)
df["is_last_hour"]  = (df.index.hour == 15).astype(int)

# FIX 1: targets in PERCENT space (×100) — Huber loss now operates on
# values like 0.15% instead of 0.0015, giving 10,000× more gradient signal
df["tgt_open_ret"]  = np.log(df["Open"].shift(-1)  / close) * RETURN_SCALE
df["tgt_close_ret"] = np.log(df["Close"].shift(-1) / close) * RETURN_SCALE
df["tgt_open_dir"]  = (df["tgt_open_ret"]  > 0).astype(int)
df["tgt_close_dir"] = (df["tgt_close_ret"] > 0).astype(int)
df["future_open"]   = df["Open"].shift(-1)
df["future_close"]  = df["Close"].shift(-1)

rows_before_drop = len(df)
df.dropna(inplace=True)
print(f"    Dropped (warm-up / shift): {rows_before_drop - len(df)}")
print(f"    Final rows: {len(df)}")

FEATURE_COLS = [c for c in df.columns if c.startswith((
    "log_", "open_log", "HL_", "OC_",
    "ret_", "RSI", "Stoch", "ROC",
    "SMA_", "EMA_", "MACD", "BB_",
    "ATR", "vol_", "OBV", "hour",
    "dow", "is_"
))]
TARGET_RET = ["tgt_open_ret", "tgt_close_ret"]
TARGET_DIR = ["tgt_open_dir", "tgt_close_dir"]
PRICE_COLS = ["future_open", "future_close"]
print(f"    Feature columns: {len(FEATURE_COLS)}")

# Compute training-set return std for clipping later
train_end_idx = int(len(df) * TRAIN_SPLIT)
train_ret_std_open  = df["tgt_open_ret"].iloc[:train_end_idx].std()
train_ret_std_close = df["tgt_close_ret"].iloc[:train_end_idx].std()
print(f"    Train return std — open: {train_ret_std_open:.4f}%  "
      f"close: {train_ret_std_close:.4f}%")


# ══════════════════════════════════════════════════════════
# 3. SPLIT + SCALE (train-only fit, no leakage)
# ══════════════════════════════════════════════════════════
print("\n[3/8] Splitting + scaling (train-only fit)...")

split = train_end_idx

scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_raw = df[FEATURE_COLS].values[:split]
X_test_raw  = df[FEATURE_COLS].values[split:]
y_ret_all   = df[TARGET_RET].values
y_dir_all   = df[TARGET_DIR].values
y_price_all = df[PRICE_COLS].values
curr_close  = df["Close"].values

scaler_X.fit(X_train_raw)
scaler_y.fit(y_ret_all[:split])

X_scaled     = np.vstack([
    scaler_X.transform(X_train_raw),
    scaler_X.transform(X_test_raw)
])
y_ret_scaled = np.vstack([
    scaler_y.transform(y_ret_all[:split]),
    scaler_y.transform(y_ret_all[split:])
])


def create_sequences(X, y_ret, y_dir, y_prices, curr_close, lookback):
    Xs, yrets, ydirs, yprices, closes = [], [], [], [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        yrets.append(y_ret[i])
        ydirs.append(y_dir[i])
        yprices.append(y_prices[i])
        closes.append(curr_close[i])
    return (np.array(Xs), np.array(yrets), np.array(ydirs),
            np.array(yprices), np.array(closes))


X_seq, y_ret_seq, y_dir_seq, y_price_seq, close_seq = create_sequences(
    X_scaled, y_ret_scaled, y_dir_all, y_price_all, curr_close, LOOKBACK
)

seq_split = split - LOOKBACK
X_train, X_test       = X_seq[:seq_split],     X_seq[seq_split:]
yret_train, yret_test = y_ret_seq[:seq_split], y_ret_seq[seq_split:]
ydir_train, ydir_test = y_dir_seq[:seq_split], y_dir_seq[seq_split:]
yprice_test           = y_price_seq[seq_split:]
close_test            = close_seq[seq_split:]

print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
n_features = X_train.shape[2]


# ══════════════════════════════════════════════════════════
# 4. WALK-FORWARD CROSS VALIDATION
# ══════════════════════════════════════════════════════════
print("\n[4/8] Walk-forward cross-validation (3 folds)...")

tscv = TimeSeriesSplit(n_splits=3)
cv_accs = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
    _m = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                        subsample=0.75, colsample_bytree=0.75,
                        random_state=SEED, tree_method="hist",
                        eval_metric="logloss")
    _m.fit(X_train[tr_idx, -1, :], ydir_train[tr_idx, 1], verbose=False)
    acc = accuracy_score(ydir_train[va_idx, 1],
                         _m.predict(X_train[va_idx, -1, :]))
    cv_accs.append(acc)
    print(f"    Fold {fold + 1}: close-direction acc = {acc * 100:.1f}%")
print(f"    CV mean = {np.mean(cv_accs)*100:.1f}%  "
      f"± {np.std(cv_accs)*100:.1f}%")


# ══════════════════════════════════════════════════════════
# 5. MODEL: Multi-Task BiLSTM
#    FIX 2: two SEPARATE direction outputs → fixes 33% training accuracy
#    FIX 5: Huber delta=0.5 in percent space → penalises near-zero harder
#    FIX 6: regression loss weight = 2.0, each dir loss = 1.0
# ══════════════════════════════════════════════════════════
print("\n[5/8] Building & training multi-task BiLSTM...")


def build_multitask_model(lookback, n_features):
    inp = Input(shape=(lookback, n_features), name="sequence_input")

    x = Bidirectional(LSTM(96, return_sequences=True))(inp)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=24)(x, x)
    x    = LayerNormalization()(Add()([x, attn]))
    x    = GlobalAveragePooling1D()(x)

    shared = Dense(64, activation="relu")(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.2)(shared)

    # Regression head (percent returns)
    reg     = Dense(32, activation="relu")(shared)
    reg_out = Dense(2, name="return_output")(reg)

    # FIX 2: separate direction heads — one per target
    # Previously both crammed into one 2-neuron output, causing one to dominate
    clf_open  = Dense(16, activation="relu")(shared)
    open_dir  = Dense(1, activation="sigmoid", name="open_dir_output")(clf_open)

    clf_close = Dense(16, activation="relu")(shared)
    close_dir = Dense(1, activation="sigmoid", name="close_dir_output")(clf_close)

    model = Model(inp, [reg_out, open_dir, close_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={
            "return_output":    tf.keras.losses.Huber(delta=0.5),  # FIX 5
            "open_dir_output":  "binary_crossentropy",
            "close_dir_output": "binary_crossentropy",
        },
        loss_weights={
            "return_output":    2.0,   # FIX 6: regression now leads
            "open_dir_output":  1.0,
            "close_dir_output": 1.0,
        },
        metrics={
            "open_dir_output":  "accuracy",
            "close_dir_output": "accuracy",
        }
    )
    return model


model = build_multitask_model(LOOKBACK, n_features)
model.summary()

callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5, monitor="val_loss")
]

history = model.fit(
    X_train,
    {
        "return_output":    yret_train,
        "open_dir_output":  ydir_train[:, 0],
        "close_dir_output": ydir_train[:, 1],
    },
    epochs=80,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)


# ══════════════════════════════════════════════════════════
# 6. XGBoost DIRECTION CLASSIFIERS
# ══════════════════════════════════════════════════════════
print("\n[6/8] Training XGBoost direction classifiers...")

X_tab_train = X_train[:, -1, :]
X_tab_test  = X_test[:, -1, :]

xgb_params = dict(
    n_estimators=500, learning_rate=0.02, max_depth=4,
    subsample=0.75, colsample_bytree=0.75,
    min_child_weight=5, gamma=0.1,
    random_state=SEED, tree_method="hist", eval_metric="logloss"
)

xgb_open_dir  = XGBClassifier(**xgb_params)
xgb_close_dir = XGBClassifier(**xgb_params)

xgb_open_dir.fit(X_tab_train,  ydir_train[:, 0],
                  eval_set=[(X_tab_test, ydir_test[:, 0])], verbose=False)
xgb_close_dir.fit(X_tab_train, ydir_train[:, 1],
                   eval_set=[(X_tab_test, ydir_test[:, 1])], verbose=False)


# ══════════════════════════════════════════════════════════
# 7. EVALUATION
# ══════════════════════════════════════════════════════════
print("\n[7/8] Evaluating...")

lstm_ret_pred, lstm_open_dir_pred, lstm_close_dir_pred = model.predict(
    X_test, verbose=0
)

# FIX 4: clip predictions to ±3σ of training return distribution
lstm_ret_inv = scaler_y.inverse_transform(lstm_ret_pred)
lstm_ret_inv[:, 0] = np.clip(lstm_ret_inv[:, 0],
                              -3 * train_ret_std_open,  3 * train_ret_std_open)
lstm_ret_inv[:, 1] = np.clip(lstm_ret_inv[:, 1],
                              -3 * train_ret_std_close, 3 * train_ret_std_close)

# Convert percent returns back to price
lstm_pred_open  = close_test * np.exp(lstm_ret_inv[:, 0] / RETURN_SCALE)
lstm_pred_close = close_test * np.exp(lstm_ret_inv[:, 1] / RETURN_SCALE)
actual_open     = yprice_test[:, 0]
actual_close    = yprice_test[:, 1]

xgb_pred_open_dir  = xgb_open_dir.predict(X_tab_test)
xgb_pred_close_dir = xgb_close_dir.predict(X_tab_test)

lstm_dir_open  = (lstm_open_dir_pred.squeeze()  > 0.5).astype(int)
lstm_dir_close = (lstm_close_dir_pred.squeeze() > 0.5).astype(int)

ensemble_open_dir  = np.where(lstm_dir_open  == xgb_pred_open_dir,
                               lstm_dir_open,  xgb_pred_open_dir)
ensemble_close_dir = np.where(lstm_dir_close == xgb_pred_close_dir,
                               lstm_dir_close, xgb_pred_close_dir)

def rmse(a, b): return np.sqrt(mean_squared_error(a, b))
def mape(a, b): return np.mean(np.abs((a - b) / (np.abs(a) + 1e-9))) * 100

actual_close_ret = np.log(actual_close / close_test)
dir_pnl = (np.sign(actual_close_ret) == (ensemble_close_dir * 2 - 1))
dir_pnl_acc = dir_pnl.mean() * 100

print("\n" + "=" * 65)
print("  PRICE PREDICTION METRICS (Test Set)")
print("=" * 65)
print(f"\n  {'Metric':<30} {'OPEN':>12} {'CLOSE':>12}")
print("  " + "-" * 55)
print(f"  {'RMSE  (₹)':<30} "
      f"{rmse(actual_open, lstm_pred_open):>12.2f} "
      f"{rmse(actual_close, lstm_pred_close):>12.2f}")
print(f"  {'MAE   (₹)':<30} "
      f"{mean_absolute_error(actual_open, lstm_pred_open):>12.2f} "
      f"{mean_absolute_error(actual_close, lstm_pred_close):>12.2f}")
print(f"  {'MAPE  (%)':<30} "
      f"{mape(actual_open, lstm_pred_open):>12.2f} "
      f"{mape(actual_close, lstm_pred_close):>12.2f}")
print(f"\n  Directional P&L accuracy (close): {dir_pnl_acc:.2f}%  "
      f"(random baseline = 50%)")

print("\n" + "=" * 65)
print("  DIRECTION ACCURACY  — random baseline = 50%")
print("=" * 65)
print(f"\n  {'Model':<30} {'OPEN DIR':>12} {'CLOSE DIR':>12}")
print("  " + "-" * 55)
print(f"  {'BiLSTM (multi-task)':<30} "
      f"{accuracy_score(ydir_test[:,0], lstm_dir_open)*100:>11.2f}% "
      f"{accuracy_score(ydir_test[:,1], lstm_dir_close)*100:>11.2f}%")
print(f"  {'XGBoost standalone':<30} "
      f"{accuracy_score(ydir_test[:,0], xgb_pred_open_dir)*100:>11.2f}% "
      f"{accuracy_score(ydir_test[:,1], xgb_pred_close_dir)*100:>11.2f}%")
print(f"  {'Ensemble':<30} "
      f"{accuracy_score(ydir_test[:,0], ensemble_open_dir)*100:>11.2f}% "
      f"{accuracy_score(ydir_test[:,1], ensemble_close_dir)*100:>11.2f}%")
print("=" * 65)

print("\n  Close Direction Detail:")
print(classification_report(ydir_test[:, 1], ensemble_close_dir,
                              target_names=["Down", "Up"]))


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════
sns.set_style("darkgrid")

# Plot 1 — Training curves (now shows BOTH direction heads separately)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].plot(history.history["loss"],     label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["open_dir_output_accuracy"],     label="Train Open Dir")
axes[1].plot(history.history["val_open_dir_output_accuracy"], label="Val Open Dir")
axes[1].axhline(0.5, color="gray", linestyle="--", label="Random")
axes[1].set_title("Open Direction Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(history.history["close_dir_output_accuracy"],     label="Train Close Dir")
axes[2].plot(history.history["val_close_dir_output_accuracy"], label="Val Close Dir")
axes[2].axhline(0.5, color="gray", linestyle="--", label="Random")
axes[2].set_title("Close Direction Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.suptitle(f"{TICKER} — Training Curves v4", fontweight="bold")
plt.tight_layout()
plt.show()

# Plot 2 — Predicted vs Actual
N        = min(150, len(actual_close))
test_idx = df.index[LOOKBACK + seq_split: LOOKBACK + seq_split + N]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, actual, pred, title in [
    (axes[0], actual_open[:N],  lstm_pred_open[:N],  "Open"),
    (axes[1], actual_close[:N], lstm_pred_close[:N], "Close"),
]:
    ax.plot(test_idx, actual, label="Actual",    color="steelblue", lw=1.5)
    ax.plot(test_idx, pred,   label="Predicted", color="tomato",    lw=1.5, linestyle="--")
    err = np.abs(actual - pred) / actual * 100
    ax.fill_between(test_idx, actual, pred, alpha=0.15, color="orange")
    ax.set_title(f"{TICKER} — {title} Price  |  Avg Error: {err.mean():.2f}%")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 3 — Feature importance
feat_imp = (pd.Series(xgb_close_dir.feature_importances_, index=FEATURE_COLS)
            .nlargest(20))
plt.figure(figsize=(10, 6))
feat_imp.sort_values().plot(kind="barh", color="steelblue", edgecolor="white")
plt.title(f"{TICKER} — XGBoost Close Direction: Top 20 Features")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════
# 8. FORECAST
#    FIX 3: ATR noise-floor prevents flatline
#           when |predicted_return| < 0.3× ATR, replace with ATR-scaled step
# ══════════════════════════════════════════════════════════
print(f"\n[8/8] Generating {FORECAST_HOURS}-hour forecast...")

BUFFER_SIZE = 50
buf = df[["Open", "High", "Low", "Close", "Volume"]].iloc[-BUFFER_SIZE:].copy()
last_atr_pct = float(df["ATR_pct"].iloc[-1])   # rolling ATR as % of price

last_time  = df.index[-1]
last_close = float(df["Close"].iloc[-1])
records    = []


def compute_features_from_buffer(buf: pd.DataFrame) -> np.ndarray:
    c = buf["Close"].squeeze()
    h = buf["High"].squeeze()
    l = buf["Low"].squeeze()
    v = buf["Volume"].squeeze()
    o = buf["Open"].squeeze()
    ts = buf.index[-1]

    row = {}
    row["log_return"]      = float(np.log(c.iloc[-1] / c.iloc[-2]))
    row["open_log_return"] = float(np.log(o.iloc[-1] / c.iloc[-2]))
    row["HL_pct"]          = float((h.iloc[-1] - l.iloc[-1]) / c.iloc[-1])
    row["OC_pct"]          = float((c.iloc[-1] - o.iloc[-1]) / o.iloc[-1])

    log_ret_series = np.log(c / c.shift(1))
    for lag in [1, 2, 3, 6, 12, 24]:
        row[f"ret_lag{lag}"] = float(log_ret_series.iloc[-lag - 1]) if len(log_ret_series) > lag else 0.0

    row["ret_mean_6"]  = float(log_ret_series.iloc[-6:].mean())
    row["ret_mean_12"] = float(log_ret_series.iloc[-12:].mean())
    row["ret_std_6"]   = float(log_ret_series.iloc[-6:].std())
    row["ret_std_12"]  = float(log_ret_series.iloc[-12:].std())

    row["RSI_14"]  = float(ta.momentum.rsi(c, window=14).iloc[-1])
    row["RSI_6"]   = float(ta.momentum.rsi(c, window=6).iloc[-1])
    _stoch         = ta.momentum.StochasticOscillator(h, l, c)
    row["Stoch_K"] = float(_stoch.stoch().iloc[-1])
    row["Stoch_D"] = float(_stoch.stoch_signal().iloc[-1])
    row["ROC_5"]   = float(ta.momentum.roc(c, window=5).iloc[-1])
    row["ROC_10"]  = float(ta.momentum.roc(c, window=10).iloc[-1])

    row["SMA_10_ratio"] = float(c.iloc[-1] / ta.trend.sma_indicator(c, window=10).iloc[-1])
    row["SMA_20_ratio"] = float(c.iloc[-1] / ta.trend.sma_indicator(c, window=20).iloc[-1])
    row["EMA_10_ratio"] = float(c.iloc[-1] / ta.trend.ema_indicator(c, window=10).iloc[-1])

    _macd             = ta.trend.MACD(c)
    row["MACD_diff"]  = float(_macd.macd_diff().iloc[-1])
    row["MACD_cross"] = int(_macd.macd().iloc[-1] > _macd.macd_signal().iloc[-1])

    _bb              = ta.volatility.BollingerBands(c)
    row["BB_pct"]    = float(_bb.bollinger_pband().iloc[-1])
    row["BB_width"]  = float((_bb.bollinger_hband().iloc[-1] -
                               _bb.bollinger_lband().iloc[-1]) / c.iloc[-1])
    row["ATR_pct"]   = float(ta.volatility.AverageTrueRange(h, l, c)
                               .average_true_range().iloc[-1] / c.iloc[-1])

    row["vol_ratio"] = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1])
    row["vol_log"]   = float(np.log1p(v.iloc[-1]))
    _obv             = ta.volume.on_balance_volume(c, v)
    _obv_med         = _obv.rolling(50).median().iloc[-1]
    row["OBV_norm"]  = float(_obv.iloc[-1] / _obv_med) if _obv_med != 0 else 1.0

    row["hour_sin"]      = float(np.sin(2 * np.pi * ts.hour / 24))
    row["hour_cos"]      = float(np.cos(2 * np.pi * ts.hour / 24))
    row["dow_sin"]       = float(np.sin(2 * np.pi * ts.weekday() / 5))
    row["dow_cos"]       = float(np.cos(2 * np.pi * ts.weekday() / 5))
    row["is_first_hour"] = int(ts.hour == 9)
    row["is_last_hour"]  = int(ts.hour == 15)

    feat_vec = np.array([row.get(f, 0.0) for f in FEATURE_COLS], dtype=np.float32)
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vec


seed_features = np.array([
    scaler_X.transform(
        compute_features_from_buffer(buf.iloc[max(0, i - BUFFER_SIZE):i + 1])
        .reshape(1, -1)
    )[0]
    for i in range(len(buf) - LOOKBACK, len(buf))
])
current_seq = seed_features.reshape(1, LOOKBACK, n_features)

while len(records) < FORECAST_HOURS:
    last_time += pd.Timedelta(hours=1)
    if not is_nse_open(last_time):
        continue

    ret_pred_s, open_dir_pred, close_dir_pred = model.predict(
        current_seq, verbose=0
    )
    ret_pred = scaler_y.inverse_transform(ret_pred_s)[0]

    # FIX 3: ATR noise-floor
    # If the predicted return magnitude is smaller than 30% of ATR,
    # the model is in lazy-zero mode — substitute direction × ATR/2 instead
    atr_floor_pct = last_atr_pct * 100 * 0.3   # 30% of hourly ATR in %

    open_ret_pct  = float(ret_pred[0])
    close_ret_pct = float(ret_pred[1])

    xgb_open_d  = int(xgb_open_dir.predict(current_seq[0, -1, :].reshape(1, -1))[0])
    xgb_close_d = int(xgb_close_dir.predict(current_seq[0, -1, :].reshape(1, -1))[0])

    if abs(open_ret_pct) < atr_floor_pct:
        sign = 1 if xgb_open_d == 1 else -1
        open_ret_pct = sign * last_atr_pct * 100 * 0.5

    if abs(close_ret_pct) < atr_floor_pct:
        sign = 1 if xgb_close_d == 1 else -1
        close_ret_pct = sign * last_atr_pct * 100 * 0.5

    # Clip to ±3σ
    open_ret_pct  = float(np.clip(open_ret_pct,
                                  -3 * train_ret_std_open,  3 * train_ret_std_open))
    close_ret_pct = float(np.clip(close_ret_pct,
                                  -3 * train_ret_std_close, 3 * train_ret_std_close))

    pred_open  = float(last_close * np.exp(open_ret_pct  / RETURN_SCALE))
    pred_close = float(last_close * np.exp(close_ret_pct / RETURN_SCALE))
    pred_high  = float(max(pred_open, pred_close) * (1 + last_atr_pct * 0.5))
    pred_low   = float(min(pred_open, pred_close) * (1 - last_atr_pct * 0.5))
    avg_vol    = float(buf["Volume"].iloc[-5:].mean())

    lstm_open_conf  = float(open_dir_pred.squeeze())
    lstm_close_conf = float(close_dir_pred.squeeze())
    open_agree  = xgb_open_d  == int(lstm_open_conf  > 0.5)
    close_agree = xgb_close_d == int(lstm_close_conf > 0.5)

    records.append({
        "Datetime":        last_time,
        "Pred_Open":       round(pred_open,  2),
        "Pred_Close":      round(pred_close, 2),
        "Open_Ret_%":      round(open_ret_pct,  4),
        "Close_Ret_%":     round(close_ret_pct, 4),
        "Open_Direction":  "▲ UP"   if xgb_open_d  == 1 else "▼ DOWN",
        "Close_Direction": "▲ UP"   if xgb_close_d == 1 else "▼ DOWN",
        "High_Confidence": open_agree and close_agree,
    })

    last_close   = pred_close
    last_atr_pct = float(np.nan_to_num(
        ta.volatility.AverageTrueRange(
            buf["High"], buf["Low"], buf["Close"]
        ).average_true_range().iloc[-1] / buf["Close"].iloc[-1],
        nan=last_atr_pct
    ))

    new_row = pd.DataFrame([{
        "Open": pred_open, "High": pred_high,
        "Low":  pred_low,  "Close": pred_close, "Volume": avg_vol,
    }], index=[last_time])
    buf = pd.concat([buf, new_row]).iloc[-BUFFER_SIZE:]

    next_feat   = compute_features_from_buffer(buf)
    next_scaled = scaler_X.transform(next_feat.reshape(1, -1))
    current_seq = np.append(current_seq[:, 1:, :],
                             next_scaled.reshape(1, 1, n_features), axis=1)


forecast_df = pd.DataFrame(records).set_index("Datetime")

daily_df = forecast_df.groupby(forecast_df.index.date).agg(
    Open=("Pred_Open",  "first"),
    Close=("Pred_Close","last"),
    Change_pct=("Pred_Close", lambda x:
        round((x.iloc[-1] /
               forecast_df.loc[forecast_df.index.date == x.index[0].date(),
                                "Pred_Open"].iloc[0] - 1) * 100, 2))
)

print("\n" + "=" * 70)
print(f"  FORECAST — {TICKER}  (Hourly)")
print("=" * 70)
print(forecast_df[["Pred_Open", "Pred_Close", "Open_Ret_%", "Close_Ret_%",
                    "Open_Direction", "Close_Direction",
                    "High_Confidence"]].to_string())

print("\n" + "=" * 60)
print(f"  FORECAST — {TICKER}  (Daily Summary)")
print("=" * 60)
print(daily_df.to_string())

# Forecast plot
fig, ax = plt.subplots(figsize=(14, 5))
context = df["Close"].iloc[-LOOKBACK * 2:]
ax.plot(context.index, context.values,
        color="steelblue", lw=1.8, label="Actual (context)")

ax.scatter(forecast_df.index, forecast_df["Pred_Open"],
           marker="o", color="orange", s=45, zorder=3, label="Pred Open")
ax.scatter(forecast_df.index, forecast_df["Pred_Close"],
           marker="s", color="tomato",  s=45, zorder=3, label="Pred Close")
ax.plot(forecast_df.index, forecast_df["Pred_Close"],
        color="tomato", lw=1.3, linestyle="--", alpha=0.8)

hc = forecast_df[forecast_df["High_Confidence"]]
if not hc.empty:
    ax.scatter(hc.index, hc["Pred_Close"],
               marker="*", color="gold", s=130, zorder=4, label="High confidence")

ax.axvline(x=df.index[-1], color="gray", linestyle=":", lw=1.5,
           label="Forecast start")
ax.set_title(f"{TICKER} — {FORECAST_HOURS}h Forecast  |  ★ High conf",
             fontweight="bold")
ax.set_ylabel("Price (₹)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Pipeline complete (v4).")
# NEXT CELL #
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════
# INTERACTIVE FORECAST PLOT (PLOTLY) - v4
# ══════════════════════════════════════════════════════════

# 1. Grab historical context
context = df["Close"].iloc[-LOOKBACK * 2:]

fig = go.Figure()

# 2. Plot Historical Actuals
fig.add_trace(go.Scatter(
    x=context.index,
    y=context.values,
    mode='lines',
    name='Actual Close (Context)',
    line=dict(color='#3498db', width=2),
    hovertemplate="Actual: ₹%{y:.2f}<extra></extra>"
))

# 3. Plot Predicted Open
fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["Pred_Open"],
    mode='markers',
    name='Predicted Open',
    marker=dict(
        color='#f39c12', 
        size=8, 
        symbol='circle',
        line=dict(width=1, color='white')
    ),
    customdata=forecast_df[["Open_Ret_%", "Open_Direction"]],
    hovertemplate="<b>Pred Open:</b> ₹%{y:.2f}<br>Return: %{customdata[0]:.2f}%<br>Dir: %{customdata[1]}<extra></extra>"
))

# 4. Plot Predicted Close (with connecting line)
fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["Pred_Close"],
    mode='lines+markers',
    name='Predicted Close',
    line=dict(color='#e74c3c', width=2, dash='dash'),
    marker=dict(
        color='#e74c3c', 
        size=8, 
        symbol='square',
        line=dict(width=1, color='white')
    ),
    customdata=forecast_df[["Close_Ret_%", "Close_Direction"]],
    hovertemplate="<b>Pred Close:</b> ₹%{y:.2f}<br>Return: %{customdata[0]:.2f}%<br>Dir: %{customdata[1]}<extra></extra>"
))

# 5. Overlay High Confidence Markers (Gold Stars)
hc = forecast_df[forecast_df["High_Confidence"]]
if not hc.empty:
    fig.add_trace(go.Scatter(
        x=hc.index,
        y=hc["Pred_Close"],
        mode='markers',
        name='High Confidence (Ensemble Agree)',
        marker=dict(
            color='#f1c40f', 
            size=16, 
            symbol='star',
            line=dict(width=1, color='white')
        ),
        hoverinfo='skip' # Skip hover so it doesn't clutter the underlying close price tooltip
    ))

# 6. Add a vertical line to mark where the forecast begins
fig.add_vline(
    x=df.index[-1].timestamp() * 1000, 
    line_width=2, 
    line_dash="dot", 
    line_color="#95a5a6",
    annotation_text="Forecast Start", 
    annotation_position="top left"
)

# 7. Apply a professional, minimalist layout
fig.update_layout(
    title=dict(
        text=f"<b>{TICKER} — {FORECAST_HOURS}h Interactive Forecast (v4)</b>",
        font=dict(size=20)
    ),
    xaxis_title="Datetime",
    yaxis_title="Price (₹)",
    template="plotly_dark",  
    hovermode="x unified",   
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=40, r=40, t=80, b=40)
)

# Render the interactive chart
fig.show()
# NEXT CELL #
