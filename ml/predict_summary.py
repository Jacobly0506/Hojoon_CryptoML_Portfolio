"""
predict_summary.py
📊 ML 예측 요약 스크립트 (15m, 1h, 4h, 1d)

- 입력: 심볼 (예: BTC, ETH)
- 처리: 각 타임프레임의 ML 예측 가격 계산
- 출력: 콘솔에 정리된 예측 결과 출력
"""

from predict_15m import predict_next_15m
from predict_1h import predict_next
from predict_4h import predict_next
from predict_1d import predict_next_1d

def main():
    symbol = input("예측할 심볼 입력 (예: BTC): ").strip().upper()

    print("\n📊 === ML 예측 요약 ===")
    print(f"📌 심볼: {symbol}")
    print("----------------------------")

    print("⏱️ 15분 예측:")
    predict_next_15m(symbol)

    print("\n⏱️ 1시간 예측:")
    predict_next(symbol)

    print("\n⏱️ 4시간 예측:")
    predict_next(symbol)

    print("\n⏱️ 1일 예측:")
    predict_next_1d(symbol)

    print("----------------------------")
    print("✅ 모든 예측 완료!")

if __name__ == "__main__":
    main()
