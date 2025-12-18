# -*- coding: utf-8 -*-
"""
BTC/USDT Daily Kline -> Feature Engineering -> RandomForest -> Evaluation -> Plots
- Download from Binance public API (no key required)
- Save raw CSV
- Build features + label
- Time-series split (no shuffle)
- Train RandomForest
- Evaluate + confusion matrix + report
- Plot: Feature Importance + True vs Predicted (direction)

Run:
  python btc_rf_full.py
"""

import time
import requests
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------
# 1) Download Binance Klines
# ---------------------------
def fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=1000, end_time_ms=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def download_history(symbol="BTCUSDT", interval="1d", total_points=3000, sleep_sec=0.2):
    """
    Download multiple pages of klines (because 1 request max 1000).
    total_points: approximate rows to keep.
    """
    all_rows = []
    end_time = None  # None means latest data

    while len(all_rows) < total_points:
        data = fetch_binance_klines(symbol=symbol, interval=interval, limit=1000, end_time_ms=end_time)
        if not data:
            break

        # prepend older chunk to the front
        all_rows = data + all_rows

        oldest_open_time = data[0][0]
        end_time = oldest_open_time - 1

        time.sleep(sleep_sec)

        # if less than 1000 returned, likely reached earliest
        if len(data) < 1000:
            break

    # keep the newest total_points
    all_rows = all_rows[-total_points:]

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


# --------------------------------
# 2) Feature Engineering + Labeling
# --------------------------------
def make_features_and_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Returns
    out["return_1d"] = out["close"].pct_change(1)
    out["return_3d"] = out["close"].pct_change(3)
    out["return_7d"] = out["close"].pct_change(7)

    # Moving averages + bias
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_bias_5"] = (out["close"] - out["ma_5"]) / out["ma_5"]

    # Volatility
    out["vol_5"] = out["return_1d"].rolling(5).std()

    # Label: tomorrow close > today close
    out["future_close"] = out["close"].shift(-1)
    out["y"] = (out["future_close"] > out["close"]).astype(int)

    # drop NaNs from rolling/pct_change/shift
    out = out.dropna().reset_index(drop=True)
    return out


# ---------------------------
# 3) Time-series split
# ---------------------------
def time_split(X, y, train_ratio=0.7):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test, split_idx


# ---------------------------
# 4) Train + Evaluate + Plot
# ---------------------------
def plot_feature_importance(model, feature_cols):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    importances.plot(kind="bar")
    plt.title("Feature Importance (Random Forest)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    return importances


def plot_true_vs_pred(plot_df):
    """
    plot_df should have: date, y_true, y_pred
    """
    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["date"], plot_df["y_true"], label="True (Actual)", alpha=0.75)
    plt.plot(plot_df["date"], plot_df["y_pred"], label="Predicted", alpha=0.75)

    plt.yticks([0, 1], ["Down", "Up"])
    plt.xlabel("Date")
    plt.ylabel("Direction")
    plt.title("BTC Direction: True vs Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # ---------- Settings ----------
    SYMBOL = "BTCUSDT"
    INTERVAL = "1d"
    TOTAL_POINTS = 3000
    RAW_CSV = "btc_daily.csv"
    TRAIN_RATIO = 0.7

    # ---------- A) Download ----------
    print("Downloading data from Binance...")
    df_raw = download_history(symbol=SYMBOL, interval=INTERVAL, total_points=TOTAL_POINTS)

    df_raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Saved raw data to: {RAW_CSV}")
    print("Raw tail sample:")
    print(df_raw.tail(3))

    # ---------- B) Feature + Label ----------
    df = make_features_and_label(df_raw)

    feature_cols = [
        "return_1d", "return_3d", "return_7d",
        "ma_5", "ma_10", "ma_bias_5",
        "vol_5"
    ]
    X = df[feature_cols]
    y = df["y"]

    # ---------- C) Time Split ----------
    X_train, X_test, y_train, y_test, split_idx = time_split(X, y, train_ratio=TRAIN_RATIO)
    print(f"\nDataset size (after dropna): {len(df)}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ---------- D) Train ----------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # ---------- E) Predict ----------
    y_pred = rf.predict(X_test)

    # ---------- F) Evaluate ----------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n=== Random Forest Result ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix [[TN FP],[FN TP]]:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # ---------- G) Plot Feature Importance ----------
    importances = plot_feature_importance(rf, feature_cols)
    print("\nFeature Importances:")
    print(importances)

    # ---------- H) Plot True vs Pred (Test Set) ----------
    plot_df = df.iloc[split_idx:].copy()
    plot_df["y_true"] = y_test.values
    plot_df["y_pred"] = y_pred
    plot_true_vs_pred(plot_df)

    # ---------- Done ----------
    print("\n🎯 Done. You now have: raw CSV + metrics + plots.")


if __name__ == "__main__":
    main()

result_df = plot_df[["date", "y_true", "y_pred"]].copy()
result_df["y_true_label"] = result_df["y_true"].map({1: "Up", 0: "Down"})
result_df["y_pred_label"] = result_df["y_pred"].map({1: "Up", 0: "Down"})

print(result_df.head(10))
