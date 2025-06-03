"""
train-4h.py
📚 Binance 4시간봉 데이터 기반 딥러닝 모델 학습 스크립트

- 입력: 심볼명 (예: BTC, ETH)
- 처리: Binance에서 4시간봉 캔들데이터 수집 → 기술적 지표 계산 → 정규화 → LSTM 모델 학습
- 출력: 학습된 모델을 models/4h_{SYMBOL}.keras 에 저장
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from services.BinanceService import BinanceService
from datetime import datetime
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

# === 시퀀스 생성 ===
def create_dataset(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len:i])
        y.append(data[i + 1][:4])  # Open, High, Low, Close 예측
    return np.array(X), np.array(y)

# === 학습 함수 ===
def train_model(symbol: str):
    print(f"📚 [{symbol}] 4시간봉 + 기술적지표 학습 시작...")

    binance = BinanceService()
    candles = binance.fetch_historical_candle_data(symbol, 1000, interval="4h")

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

    data = np.array(data)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    data_scaled = (data - min_vals) / (max_vals - min_vals + 1e-8)

    seq_len = 60
    X, y = create_dataset(data_scaled, seq_len)

    model = Sequential([
        Input(shape=(seq_len, X.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(4)  # 다중 출력 (Open, High, Low, Close)
    ])

    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(patience=100, restore_best_weights=True)
    model.fit(X, y, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stop])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("models", exist_ok=True)
    model_path = f"models/4h_{symbol}_{timestamp}.keras"
    model.save(model_path)
    print(f"✅ 모델 저장 완료: {model_path}")

# === 실행 ===
if __name__ == "__main__":
    symbol = input("학습할 심볼 입력 (예: BTC): ").strip().upper()
    train_model(symbol)