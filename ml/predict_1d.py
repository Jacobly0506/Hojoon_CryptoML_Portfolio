"""
predict-1d.py
🔮 Binance 1일봉 데이터 기반 ML 예측 스크립트 (OHLC 예측 버전)

- 입력: 심볼명 (예: BTC, ETH)
- 처리: 최신 1일봉 데이터 수집 → 기술적 지표 계산 → 정규화 → LSTM 모델 예측
- 출력: 다음 1일의 예측 Open, High, Low, Close 가격 출력
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
from services.BinanceService import BinanceService

# === 기술적 지표 계산 함수 ===
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

def calculate_macd(prices, short=12, long=26):
    if len(prices) < long:
        return np.nan
    return calculate_ema(prices, short) - calculate_ema(prices, long)

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

# === 최신 모델 불러오기 ===
def get_latest_model_path(symbol, model_dir="models"):
    files = [f for f in os.listdir(model_dir) if f.startswith(f"1d_{symbol}_") and f.endswith(".keras")]
    if not files:
        return None
    latest_model = max(files, key=lambda x: x.split(f"1d_{symbol}_")[-1].replace(".keras", ""))
    return os.path.join(model_dir, latest_model)

# === 예측 함수 ===
def predict_next_1d(symbol: str):
    model_path = get_latest_model_path(symbol)
    if model_path is None:
        print(f"❌ 모델이 없습니다: models/1d_{symbol}_*.keras")
        return

    model = load_model(model_path)
    binance = BinanceService()
    candles = binance.fetch_historical_candle_data(symbol, 1000, interval="1d")

    closes = [c.close for c in candles]
    data = []
    for i, c in enumerate(candles):
        sma = calculate_sma(closes[:i+1], 20)
        ema = calculate_ema(closes[:i+1], 20)
        rsi = calculate_rsi(closes[:i+1])
        macd = calculate_macd(closes[:i+1])
        atr = calculate_atr(candles[:i+1])
        obv = calculate_obv(candles[:i+1])
        if any(np.isnan(x) for x in [sma, ema, rsi, macd, atr, obv]):
            continue
        row = [c.open, c.high, c.low, c.close, sma, ema, rsi, macd, atr, obv]
        data.append(row)

    if len(data) < 60:
        print("❌ 예측에 필요한 데이터 부족 (최소 60개 필요)")
        return

    recent_data = np.array(data[-60:])
    min_vals, max_vals = recent_data.min(axis=0), recent_data.max(axis=0)
    scaled = (recent_data - min_vals) / (max_vals - min_vals + 1e-8)

    X = scaled.reshape((1, 60, scaled.shape[1]))
    pred_scaled = model.predict(X)[0]  # [Open, High, Low, Close] 스케일값

    predicted_ohlc = pred_scaled * (max_vals[:4] - min_vals[:4] + 1e-8) + min_vals[:4]
    predicted_open, predicted_high, predicted_low, predicted_close = predicted_ohlc

    print(f"🔮 [{symbol}] 다음 1일 예측 OHLC:")
    print(f"    Open : {predicted_open:.2f} USD")
    print(f"    Close: {predicted_close:.2f} USD")
    print(f"    High : {predicted_high:.2f} USD")
    print(f"    Low  : {predicted_low:.2f} USD")

# === 실행 ===
if __name__ == "__main__":
    symbol_input = input("예측할 심볼 입력 (예: BTC): ").strip().upper()
    predict_next_1d(symbol_input)