"""
train_summary.py
📚 ML 학습 요약 스크립트 (15m, 1h, 4h, 1d) - 병렬 처리 전용 (전체 학습)

- 입력: 심볼 (예: BTC, ETH)
- 처리: 각 타임프레임 모델 학습을 병렬 실행
- 출력: 콘솔에 정리된 학습 결과 출력
"""

from train_15m import train_model as train_15m
from train_1h import train_model as train_1h
from train_4h import train_model as train_4h
from train_1d import train_model as train_1d
from multiprocessing import Process

def run_in_process(func, symbol):
    p = Process(target=func, args=(symbol,))
    p.start()
    return p

def main():
    symbol = input("📥 학습할 심볼 입력 (예: BTC): ").strip().upper()

    print("\n📚 === ML 전체 학습 시작 (병렬 실행) ===")
    print(f"📌 심볼: {symbol}")
    print("----------------------------")

    # 병렬 실행
    processes = [
        run_in_process(train_15m, symbol),
        run_in_process(train_1h, symbol),
        run_in_process(train_4h, symbol),
        run_in_process(train_1d, symbol),
    ]

    # 학습 완료까지 대기
    for p in processes:
        p.join()

    print("----------------------------")
    print("✅ 모든 학습 완료!")

if __name__ == "__main__":
    main()