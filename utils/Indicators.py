# utils/indicators.py

def calculate_sma(prices: list[float], period: int) -> float:
    if len(prices) < period:
        return -1
    return sum(prices[-period:]) / period

def calculate_ema(prices: list[float], period: int) -> float:
    if len(prices) < period:
        return -1
    k = 2 / (period + 1)
    ema = prices[-period]
    for price in prices[-period + 1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return -1
    gains = 0
    losses = 0
    for i in range(-period, 0):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_vwap(price_volume: list[list[float]]) -> float:
    total_pv = 0
    total_volume = 0
    for price, volume in price_volume:
        total_pv += price * volume
        total_volume += volume
    return total_pv / total_volume if total_volume != 0 else -1

def calculate_atr(candles: list, period: int = 14) -> float:
    if len(candles) < period + 1:
        return -1
    trs = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs[-period:]) / period

def calculate_obv(candles: list) -> float:
    if not candles:
        return -1
    obv = 0
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv += candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            obv -= candles[i].volume
    return obv

def calculate_macd(prices: list[float], short: int = 12, long: int = 26) -> float:
    if len(prices) < long:
        return -1
    short_ema = calculate_ema(prices, short)
    long_ema = calculate_ema(prices, long)
    return short_ema - long_ema

def calculate_maci(prices: list[float], short: int = 12, long: int = 26, signal: int = 9) -> float:
    if len(prices) < long + signal:
        return -1

    macd_line = []
    for i in range(long, len(prices)):
        short_ema = calculate_ema(prices[:i+1], short)
        long_ema = calculate_ema(prices[:i+1], long)
        macd = short_ema - long_ema
        macd_line.append(macd)

    signal_ema = calculate_ema(macd_line, signal)
    return macd_line[-1] - signal_ema  # MACI = MACD - Signal