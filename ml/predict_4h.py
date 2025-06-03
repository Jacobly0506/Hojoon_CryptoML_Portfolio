"""
predict-4h.py
🔮 Binance 4시간봉 데이터 기반 ML 예측 스크립트 (OHLC 예측)

- 입력: 심볼명 (예: BTC, ETH)
- 처리: 최신 4시간봉 데이터 수집 → 기술적 지표 계산 → 정규화 → LSTM 모델 예측
- 출력: 다음 4시간의 예측 OHLC (Open, High, Low, Close) 출력
"""

import numpy as np
from tensorflow.keras.models import load_model
from services.BinanceService import BinanceService
import os

# === 기술적 지표 계산 ===
def calculate_sma(prices, period):
    return np.mean(prices[-period:]) if len(prices) >= period else np.nan

def calculate_ema(prices, period):
    if len(prices) < period:
        return np.nan
    k = 2 / (period + 1)
    ema = prices[-period]
    for price in prices[-period + 1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return np.nan
    gains, losses = 0, 0
    for i in range(-period, 0):
        diff = prices[i] - prices[i - 1]
        gains += max(0, diff)
        losses += max(0, -diff)
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_period=12, long_period=26):
    if len(prices) < long_period:
        return np.nan
    return calculate_ema(prices, short_period) - calculate_ema(prices, long_period)

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return np.nan
    trs = []
    for i in range(-period, 0):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return np.mean(trs)

def calculate_obv(candles):
    obv = 0
    prev_close = candles[0].close
    for c in candles[1:]:
        if c.close > prev_close:
            obv += c.volume
        elif c.close < prev_close:
            obv -= c.volume
        prev_close = c.close
    return obv

# === 예측 함수 ===
def predict_next(symbol: str):
    model_dir = "models"
    available_models = [f for f in os.listdir(model_dir) if f.startswith(f"4h_{symbol}_") and f.endswith(".keras")]
    if not available_models:
        print(f"❌ 모델이 없습니다: models/4h_{symbol}_*.keras")
        return

    latest_model = sorted(available_models)[-1]
    model_path = os.path.join(model_dir, latest_model)
    model = load_model(model_path)

    binance = BinanceService()
    candles = binance.fetch_historical_candle_data(symbol, 1000, "4h")

    closes = [c.close for c in candles]
    data = []
    for i, c in enumerate(candles):
        sma = calculate_sma(closes[:i + 1], 20)
        ema = calculate_ema(closes[:i + 1], 20)
        rsi = calculate_rsi(closes[:i + 1])
        macd = calculate_macd(closes[:i + 1])
        atr = calculate_atr(candles[:i + 1])
        obv = calculate_obv(candles[:i + 1])

        if any(np.isnan(val) for val in [sma, ema, rsi, macd, atr, obv]):
            continue

        row = [c.open, c.high, c.low, c.close, sma, ema, rsi, macd, atr, obv]
        data.append(row)

    if len(data) < 60:
        print("❌ 예측에 필요한 데이터가 부족합니다. 최소 60개의 유효 데이터 필요")
        return

    recent_data = np.array(data[-60:])
    min_vals = recent_data.min(axis=0)
    max_vals = recent_data.max(axis=0)
    data_scaled = (recent_data - min_vals) / (max_vals - min_vals + 1e-8)

    X = data_scaled.reshape((1, 60, recent_data.shape[1]))  # (1, 60, 10)

    prediction_scaled = model.predict(X)[0]  # shape = (4,)
    predicted_ohlc = prediction_scaled * (max_vals[:4] - min_vals[:4] + 1e-8) + min_vals[:4]

    print(f"🔮 [{symbol}] 다음 4시간 예측 OHLC:")
    print(f"   ▸ Open : {predicted_ohlc[0]:.2f} USD")
    print(f"   ▸ Close: {predicted_ohlc[3]:.2f} USD")
    print(f"   ▸ High : {predicted_ohlc[1]:.2f} USD")
    print(f"   ▸ Low  : {predicted_ohlc[2]:.2f} USD")

# === 실행 ===
if __name__ == "__main__":
    symbol_input = input("예측할 심볼 입력 (예: BTC): ").strip().upper()
    predict_next(symbol_input)