from services.BinanceService import BinanceService
from utils.Indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_vwap, calculate_atr, calculate_obv,
    calculate_macd, calculate_maci
)

def run_full_analysis(symbol: str, timeframe: str):
    binance = BinanceService()

    print(f"\n📊 분석 시작: {symbol} ({timeframe})")
    print(f"==========================")

    # 현재가
    price_info = binance.fetch_price(symbol)
    print(f"현재가 (Binance): {float(price_info.price):.2f} USD")
    print("------")

    # 과거 가격 데이터
    prices = binance.fetch_historical_prices(symbol, 400, timeframe)
    print("SMA:")
    for p in [7, 15, 50, 200, 400]:
        sma = calculate_sma(prices, p)
        if sma != -1:
            comment = "(단기 평균)" if p == 7 else "(중기 추세)" if p == 15 else "(장기 추세)"
            print(f" - {p}-period SMA: {sma:.2f} {comment}")
        else:
            print(f" - {p}-period SMA: N/A")

    print("------")
    print("RSI:")
    rsi = calculate_rsi(prices, 14)
    if rsi != -1:
        signal = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
        print(f" - 14-period RSI: {rsi:.2f} (📌 < 30 과매도 / > 70 과매수 → {signal})")
    else:
        print(" - Not enough data")

    print("------")
    print("VWAP:")
    price_volume = binance.fetch_historical_price_volume(symbol, 400, timeframe)
    vwap = calculate_vwap(price_volume)
    if vwap != -1:
        position = "매수 우위" if float(price_info.price) > vwap else "매도 우위"
        print(f" - VWAP (400): {vwap:.2f} (📌 현재가 기준: {position})")
    else:
        print(" - Not enough data")

    print("------")
    print("ATR & OBV:")
    candles = binance.fetch_historical_candle_data(symbol, 100, timeframe)
    atr = calculate_atr(candles, 14)
    obv = calculate_obv(candles)
    print(f" - ATR (14): {atr:.2f} (📌 변동성 지표, 수치가 클수록 가격 출렁임 ↑)" if atr != -1 else " - ATR: N/A")
    print(f" - OBV: {obv:.2f} (📌 OBV 증가: 매수세 / 감소: 매도세)" if obv != -1 else " - OBV: N/A")

    print("------")
    print("MACD & MACI:")
    macd = calculate_macd(prices)
    maci = calculate_maci(prices)
    print(f" - MACD: {macd:.4f} (📌 양수: 상승 추세 / 음수: 하락 추세 가능성)" if macd != -1 else " - MACD: N/A")
    print(f" - MACI: {maci:.4f} (📌 MACD - Signal → 음수: 약세 전환 시그널)" if maci != -1 else " - MACI: N/A")

    print("==========================")


def main():
    while True:
        print("\n===== 메뉴 =====")
        print("1. 분석 실행")
        print("2. 종료")
        choice = input("입력 (1~2): ").strip()

        if choice == "1":
            symbol = input("📥 심볼 입력 (예: BTC): ").strip().upper()
            timeframe = input("⏱️ 봉 구간 (15m / 1h / 4h / 1d): ").strip()
            run_full_analysis(symbol, timeframe)
        elif choice == "2" or choice.lower() == "exit":
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 입력입니다.")

if __name__ == "__main__":
    main()
