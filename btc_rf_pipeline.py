# -*- coding: utf-8 -*-
"""
BTC/USDT Kline -> Feature Engineering -> RandomForest -> Evaluation
Support multi-interval (1h/2h/4h/1d) + multi-horizon prediction (N bars ahead)
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
    end_time = None  # None means latest

    while len(all_rows) < total_points:
        data = fetch_binance_klines(symbol=symbol, interval=interval, limit=1000, end_time_ms=end_time)
        if not data:
            break

        all_rows = data + all_rows  # prepend older chunk
        oldest_open_time = data[0][0]
        end_time = oldest_open_time - 1

        time.sleep(sleep_sec)

        if len(data) < 1000:
            break

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
def make_features_and_label(df: pd.DataFrame, horizon: int, windows=(5, 10, 5)):
    """
    horizon: predict N bars ahead (N candles later)
      - if interval = 1h, horizon=2 => predict 2 hours ahead
      - if interval = 4h, horizon=3 => predict 12 hours ahead

    windows: (ma_short, ma_long, vol_window)
    """
    ma_short, ma_long, vol_w = windows
    out = df.copy()

    # Returns (still in "bars" units)
    out["return_1"] = out["close"].pct_change(1)
    out["return_3"] = out["close"].pct_change(3)
    out["return_7"] = out["close"].pct_change(7)

    # Moving averages + bias
    out["ma_s"] = out["close"].rolling(ma_short).mean()
    out["ma_l"] = out["close"].rolling(ma_long).mean()
    out["ma_bias_s"] = (out["close"] - out["ma_s"]) / out["ma_s"]

    # Volatility
    out["vol"] = out["return_1"].rolling(vol_w).std()

    # Label: N bars later close > now close
    out["future_close"] = out["close"].shift(-horizon)
    out["y"] = (out["future_close"] > out["close"]).astype(int)

    out = out.dropna().reset_index(drop=True)
    return out


# ---------------------------
# 3) Train / Test split (time)
# ---------------------------
def time_split(X: pd.DataFrame, y: pd.Series, train_ratio=0.7):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test, split_idx


# ---------------------------
# 4) Train + Evaluate + Plot
# ---------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("=== Random Forest Result ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix [[TN FP],[FN TP]]:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances)

    plt.figure(figsize=(10, 4))
    importances.plot(kind="bar")
    plt.title("Feature Importance (Random Forest)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    return rf, y_pred


def main():
    # ---------- Settings ----------
    SYMBOL = "BTCUSDT"

    # ✅ 你要的時間尺度：改這裡
    INTERVAL = "1h"      # "1h", "2h", "4h", "1d"

    # ✅ 你要預測多久後：這是 "N 根 K 線後"
    HORIZON = 2          # 若 INTERVAL="1h"，HORIZON=2 => 預測2小時後方向

    TOTAL_POINTS = 3000
    TRAIN_RATIO = 0.7

    OUTPUT_CSV = f"btc_{INTERVAL}.csv"

    # 可調特徵 window（以 bar 為單位）
    # 1h 的話：ma 5/10 = 5h/10h；4h 的話：5/10 = 20h/40h
    MA_SHORT = 5
    MA_LONG = 10
    VOL_W = 5

    print(f"\n[Config] interval={INTERVAL}, horizon={HORIZON} bars ahead")
    print(f"[Config] windows: ma_short={MA_SHORT}, ma_long={MA_LONG}, vol={VOL_W}")

    # ---------- A) Download ----------
    print("Downloading data from Binance...")
    df_raw = download_history(symbol=SYMBOL, interval=INTERVAL, total_points=TOTAL_POINTS)

    df_raw.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Saved raw data to: {OUTPUT_CSV}")
    print(df_raw.tail(3))

    # ---------- B) Feature + Label ----------
    df = make_features_and_label(df_raw, horizon=HORIZON, windows=(MA_SHORT, MA_LONG, VOL_W))

    feature_cols = ["return_1", "return_3", "return_7", "ma_s", "ma_l", "ma_bias_s", "vol"]
    X = df[feature_cols]
    y = df["y"]

    # ---------- C) Time split ----------
    X_train, X_test, y_train, y_test, split_idx = time_split(X, y, train_ratio=TRAIN_RATIO)
    print(f"\nDataset size (after dropna): {len(df)}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ---------- D) Train + Evaluate ----------
    model, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)

    # ---------- E) 你要的：看每一天/每一根K到底喊 Up or Down ----------
    result_df = df.iloc[split_idx:][["date", "close"]].copy()
    result_df["y_true"] = y_test.values
    result_df["y_pred"] = y_pred
    result_df["true_label"] = result_df["y_true"].map({1: "Up", 0: "Down"})
    result_df["pred_label"] = result_df["y_pred"].map({1: "Up", 0: "Down"})

    print("\n=== True vs Pred (first 10 rows in test set) ===")
    print(result_df.head(10))

    last = result_df.iloc[-1]
    print("\n=== Latest prediction (based on last available candle) ===")
    print("Feature time:", last["date"])
    print(f"Predicted direction for next {HORIZON} bars:", last["pred_label"])

    print("\n🎯 Done.")


if __name__ == "__main__":
    main()
