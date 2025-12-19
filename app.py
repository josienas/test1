import mlflow
import mlflow.sklearn
import time
import math
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 設定 MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

st.set_page_config(page_title="BTC Direction Predictor", layout="wide")
st.title("BTC/USDT 漲跌方向預測")

#666666
# ---------------------------
# Binance download
# ---------------------------
def fetch_binance_klines(symbol="BTCUSDT", interval="1d", limit=1000, end_time_ms=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def download_history(symbol="BTCUSDT", interval="1d", total_points=3000, sleep_sec=0.15):
    all_rows = []
    end_time = None
    while len(all_rows) < total_points:
        data = fetch_binance_klines(symbol=symbol, interval=interval, limit=1000, end_time_ms=end_time)
        if not data:
            break
        all_rows = data + all_rows
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
    df = df.drop_duplicates(subset="date")
    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_csv_by_interval(interval: str, total_points: int):
    """
    Read local CSV file:
      btc_1h.csv / btc_2h.csv / btc_4h.csv / btc_1d.csv
    Accepts either:
      - columns: open_time, open, high, low, close, volume
      - or columns: date, open, high, low, close, volume
    """
    path = f"btc_{interval}.csv"
    df = pd.read_csv(path)

    # normalize datetime column
    if "open_time" in df.columns:
        df["date"] = pd.to_datetime(df["open_time"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"{path} 缺少 open_time 或 date 欄位")

    # normalize numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset="date")
    df = df.sort_values("date").reset_index(drop=True)

    if total_points < len(df):
        df = df.tail(total_points).reset_index(drop=True)

    return df


# ---------------------------
# Feature + label
# ---------------------------
def make_features_and_label(df: pd.DataFrame, horizon_bars: int, ma_short=5, ma_long=10, vol_w=5) -> pd.DataFrame:
    out = df.copy()

    # ---------- Basic returns ----------
    out["return_1"] = out["close"].pct_change(1)
    out["return_3"] = out["close"].pct_change(3)
    out["return_7"] = out["close"].pct_change(7)

    # ---------- Price range / candle structure ----------
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]                 # (H-L)/C
    out["oc_return"] = (out["close"] - out["open"]) / out["open"]               # (C-O)/O
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["close"]
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["close"]

    # ---------- Volume features ----------
    out["vol_chg_1"] = out["volume"].pct_change(1)
    out["vol_ma_20"] = out["volume"].rolling(20).mean()
    out["vol_ratio_20"] = out["volume"] / out["vol_ma_20"]

    # ---------- Moving averages + bias ----------
    out["ma_s"] = out["close"].rolling(ma_short).mean()
    out["ma_l"] = out["close"].rolling(ma_long).mean()
    out["ma_bias_s"] = (out["close"] - out["ma_s"]) / out["ma_s"]
    out["ma_bias_l"] = (out["close"] - out["ma_l"]) / out["ma_l"]

    # ---------- Volatility ----------
    out["vol"] = out["return_1"].rolling(vol_w).std()

    # ---------- RSI(14) ----------
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ---------- MACD(12,26,9) ----------
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # ---------- Bollinger Bands(20,2) ----------
    bb_mid = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std()
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (out["bb_mid"] + 1e-12)
    out["bb_pos"] = (out["close"] - out["bb_lower"]) / ((out["bb_upper"] - out["bb_lower"]) + 1e-12)

    # ---------- ATR(14) ----------
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_pct_14"] = out["atr_14"] / (out["close"] + 1e-12)

    # ---------- Label: N bars later close > now close ----------
    out["future_close"] = out["close"].shift(-horizon_bars)
    out["y"] = (out["future_close"] > out["close"]).astype(int)

    return out.dropna().reset_index(drop=True)


def interval_to_hours(interval: str) -> int:
    mapping = {"1h": 1, "2h": 2, "4h": 4, "1d": 24}
    return mapping[interval]

def _ensure_enough_rows(df: pd.DataFrame, horizon_bars: int, ma_long: int, vol_w: int) -> int:
    """Return minimum required rows; raise Streamlit stop if insufficient."""
    min_lookback = max(ma_long, vol_w, 26, 20, 14)  # longest window used in features
    min_rows = min_lookback + horizon_bars + 2  # buffer to allow dropna and splitting
    if len(df) < min_rows:
        st.error(
            f"資料不足（目前 {len(df)} 筆，至少需要 {min_rows} 筆）才能計算特徵與分割訓練/測試集，"
            f"請減少 horizon 或增加資料筆數。"
        )
        st.stop()
    return min_rows


def _safe_train_test_split(X: pd.DataFrame, y: pd.Series, train_ratio: float):
    split_idx = int(len(X) * train_ratio)
    split_idx = min(max(split_idx, 1), len(X) - 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if X_train.empty or X_test.empty:
        st.error("資料分割後訓練/測試集有空值，請調整 train ratio 或資料筆數。")
        st.stop()
    return X_train, X_test, y_train, y_test, split_idx


# ---------------------------
# UI Controls
# ---------------------------
st.sidebar.header("設定")

data_source = st.sidebar.radio("資料來源", ["抓 Binance（即時）", "讀本機 CSV（btc_1h.csv...）"], index=0)

interval = st.sidebar.selectbox("K 線時間尺度", ["1h", "2h", "4h", "1d"], index=3)  # default 1d
total_points = st.sidebar.slider("資料筆數（越大越久）", 500, 5000, 3000, 500)

horizon_hours = st.sidebar.selectbox("預測多久後（小時）", [1, 2, 4, 8, 12, 24], index=1)

train_ratio = st.sidebar.slider("Train ratio", 0.5, 0.9, 0.7, 0.05)
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
max_depth = st.sidebar.slider("max_depth", 2, 20, 5, 1)

seed = st.sidebar.number_input("random_state（seed）", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
# MA sliders (hidden)
# ma_short = st.sidebar.slider("MA short（bars）", 3, 60, 5, 1)
# ma_long = st.sidebar.slider("MA long（bars）", 5, 120, 10, 1)
# vol_w = st.sidebar.slider("Vol window（bars）", 3, 60, 5, 1)
ma_short = 5
ma_long = 10
vol_w = 5

# ✅ 強制刷新按鈕（清 cache + rerun）
if st.sidebar.button("🔄 強制重新抓資料 / 重訓"):
    st.cache_data.clear()
    st.rerun()

# horizon bars (use ceil)
bar_hours = interval_to_hours(interval)
horizon_bars = max(1, math.ceil(horizon_hours / bar_hours))

st.info(
    f"你選的是：interval = {interval}（每根 {bar_hours} 小時）｜"
    f"預測 = {horizon_hours} 小時後 ≈ {horizon_bars} 根K 後"
)


# ---------------------------
# Load data (Binance or CSV)
# ---------------------------
if data_source.startswith("抓 Binance"):
    with st.spinner("抓取 Binance 資料中..."):
        df_raw = download_history(symbol="BTCUSDT", interval=interval, total_points=total_points)
else:
    with st.spinner("讀取本機 CSV 中..."):
        # 需要你資料夾內有 btc_1h.csv / btc_2h.csv / btc_4h.csv / btc_1d.csv
        df_raw = load_csv_by_interval(interval=interval, total_points=total_points)

df_raw = df_raw.copy()
df_raw = df_raw.dropna(subset=["date", "close"])


# ✅ 驗證：你真的換到 interval 了嗎？
st.subheader("資料檢查")
c1, c2, c3 = st.columns(3)
c1.write("資料筆數")
c1.metric("rows", f"{len(df_raw)}")

c2.write("時間範圍")
c2.write(f"{df_raw['date'].min()}  ~  {df_raw['date'].max()}")

c3.write("最後兩筆時間差")
if len(df_raw) >= 2:
    c3.write(df_raw["date"].iloc[-1] - df_raw["date"].iloc[-2])
else:
    c3.write("資料不足")

st.subheader("原始資料（最後 10 筆）")
st.dataframe(df_raw.tail(10), use_container_width=True)


# ---------------------------
# Build dataset
# ---------------------------
_ensure_enough_rows(df_raw, horizon_bars=horizon_bars, ma_long=ma_long, vol_w=vol_w)

df = make_features_and_label(df_raw, horizon_bars=horizon_bars, ma_short=ma_short, ma_long=ma_long, vol_w=vol_w)

if df.empty:
    st.error("特徵計算後沒有可用資料，請調整參數或增加資料筆數。")
    st.stop()


feature_cols = [
    # returns
    "return_1", "return_3", "return_7",

    # candle / range
    "hl_range", "oc_return", "upper_wick", "lower_wick",

    # volume
    "vol_chg_1", "vol_ratio_20",

    # MA + bias
    "ma_s", "ma_l", "ma_bias_s", "ma_bias_l",

    # volatility
    "vol",

    # RSI / MACD
    "rsi_14", "macd", "macd_signal", "macd_hist",

    # Bollinger / ATR
    "bb_width", "bb_pos", "atr_pct_14",
]

X = df[feature_cols]
y = df["y"]

X_train, X_test, y_train, y_test, split_idx = _safe_train_test_split(X, y, train_ratio)

# ---------------------------
# Train + predict
# ---------------------------
# 確保結束任何前一個 run
mlflow.end_run()

with mlflow.start_run(nested=False):
    # 記錄參數
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("train_ratio", train_ratio)
    mlflow.log_param("horizon_hours", horizon_hours)
    mlflow.log_param("interval", interval)

    # 訓練
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 評估
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # 存模型
    mlflow.sklearn.log_model(rf, "random_forest_model")
    
baseline = max((y_test == 1).mean(), (y_test == 0).mean())

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Test Accuracy", f"{acc:.6f}")
col2.metric("Majority baseline", f"{baseline:.6f}")
col3.metric("Train size", f"{len(X_train)}")
col4.metric("Test size", f"{len(X_test)}")
col5.metric("Pred Up ratio", f"{(y_pred == 1).mean():.3f}")


# 最新預測
latest_features_time = df.iloc[-1]["date"]
latest_pred = rf.predict(X.iloc[[-1]])[0]
latest_label = "Up（預測會漲）" if latest_pred == 1 else "Down（預測會跌/不漲）"
st.success(f"📌 最新特徵時間（{latest_features_time}）→ 模型預測 **{horizon_hours} 小時後**：**{latest_label}**")


# ---------------------------
# Eval (hidden)
# ---------------------------
# st.subheader("混淆矩陣 / 報告")
# cm = confusion_matrix(y_test, y_pred)
# st.write("Confusion Matrix [[TN FP],[FN TP]]:")
# st.write(cm)
# st.text(classification_report(y_test, y_pred, digits=4))


# ---------------------------
# Feature importance plot (hidden)
# ---------------------------
# st.subheader("特徵重要性")
# importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
# fig1 = plt.figure()
# importances.plot(kind="bar")
# plt.title("Feature Importance")
# plt.tight_layout()
# st.pyplot(fig1)


# ---------------------------
# True vs Pred plot (hidden)
# ---------------------------
# st.subheader("True vs Pred（測試集方向）")
# plot_df = df.iloc[split_idx:].copy()
# plot_df["y_true"] = y_test.values
# plot_df["y_pred"] = y_pred
# 
# fig2 = plt.figure(figsize=(12, 3.5))
# plt.plot(plot_df["date"], plot_df["y_true"], label="True", alpha=0.75)
# plt.plot(plot_df["date"], plot_df["y_pred"], label="Pred", alpha=0.75)
# plt.yticks([0, 1], ["Down", "Up"])
# plt.title("True vs Pred (Test Set)")
# plt.legend()
# plt.tight_layout()
# st.pyplot(fig2)


# ---------------------------
# Show a few predictions table
# ---------------------------
st.subheader("測試集前 15 筆：真實 vs 預測")
# 因為 plot_df 被定義在被註解的區段中，需要在這裡重新定義
plot_df = df.iloc[split_idx:].copy()
plot_df["y_true"] = y_test.values
plot_df["y_pred"] = y_pred

show_df = plot_df[["date", "close", "y_true", "y_pred"]].head(15).copy()
show_df["y_true_label"] = show_df["y_true"].map({1: "Up", 0: "Down"})
show_df["y_pred_label"] = show_df["y_pred"].map({1: "Up", 0: "Down"})
st.dataframe(show_df, use_container_width=True)

