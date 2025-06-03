import os
import time
import csv
from datetime import datetime
from services.BinanceService import BinanceService

# CSV 경로 생성
def get_csv_path(symbol: str) -> str:
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{symbol.upper()}_1m.csv")

# CSV 수집 함수
def collect_5m(symbol: str, csv_path: str):
    binance = BinanceService()

    # 가장 최근 timestamp 확인
    last_timestamp = 0
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                last_timestamp = int(rows[-1][0])

    # Binance에서 1분봉 800개 가져오기
    candles = binance.fetch_historical_candle_data(symbol, 800, "1m")

    new_rows = []
    for i, c in enumerate(candles):
        # timestamp 직접 계산 (가장 오래된 것이 먼저)
        timestamp = int((time.time() // 60 - (len(candles) - i)) * 60 * 1000)
        if timestamp > last_timestamp:
            new_rows.append([
                timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume
            ])

    if not new_rows:
        print("📭 저장할 새로운 데이터가 없습니다.")
        return

    # CSV에 누적 저장
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"✅ {len(new_rows)}개의 1분봉 데이터 저장 완료 ({symbol.upper()})")

# 실행 진입점
def main():
    symbol = input("📥 심볼 입력 (예: BTC): ").strip().upper()
    csv_path = get_csv_path(symbol)

    # 주기적으로 수집
    while True:
        try:
            collect_5m(symbol, csv_path)
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
        time.sleep(60)  # 1분마다 실행

if __name__ == "__main__":
    main()
