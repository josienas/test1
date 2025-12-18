import requests
import pandas as pd
import time

url = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"

intervals = ["1h", "2h", "4h", "1d"]

cols = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume",
    "num_trades", "taker_buy_base_vol",
    "taker_buy_quote_vol", "ignore"
]

for interval in intervals:
    print(f"📥 Downloading {interval} data...")

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    filename = f"btc_{interval}.csv"
    df.to_csv(filename, index=False)

    print(f"✅ {interval} 完成，共 {len(df)} 筆 → {filename}")

    time.sleep(0.2)  # 友善一點，避免打太快
