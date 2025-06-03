import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

# ===== 설정 =====
SEQ_LEN = 60
EPOCHS = 100
BATCH_SIZE = 32
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== 시퀀스 생성 함수 =====
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# ===== 가장 최근 모델 불러오기 =====
def get_latest_model_path(symbol: str):
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{symbol.upper()}_1m")]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(MODEL_DIR, files[0])

# ===== 모델 구성 함수 =====
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(4)  # open, high, low, close
    ])
    model.compile(loss="mse", optimizer="adam")
    return model

# ===== 학습 함수 =====
def train_model(symbol: str, incremental=False):
    csv_path = f"data/{symbol.upper()}_1m.csv"
    if not os.path.exists(csv_path):
        print("❌ CSV 파일이 없습니다. 먼저 데이터를 수집해주세요.")
        return

    df = pd.read_csv(csv_path, header=None)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.dropna()

    features = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = create_sequences(scaled_data, SEQ_LEN)
    y = y[:, :4]  # OHLC만 예측

    if incremental:
        # 가장 최근 모델 로드
        latest_model_path = get_latest_model_path(symbol)
        if latest_model_path:
            print(f"📂 기존 모델 로드: {latest_model_path}")
            model = load_model(latest_model_path)
        else:
            print("⚠️ 기존 모델이 없어 새로 생성합니다.")
            model = build_model((SEQ_LEN, len(features)))

        # 최근 데이터로만 훈련
        X_recent = X[-BATCH_SIZE:]
        y_recent = y[-BATCH_SIZE:]
        model.fit(X_recent, y_recent, epochs=1, verbose=1)

        # 덮어쓰기
        model.save(latest_model_path)
        print(f"🔄 모델 업데이트 완료: {latest_model_path}")

    else:
        # 새 모델 생성 및 전체 학습
        model = build_model((SEQ_LEN, len(features)))
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = os.path.join(MODEL_DIR, f"{symbol.upper()}_1m_{timestamp}.keras")
        model.save(model_path)
        print(f"✅ 모델 저장 완료: {model_path}")

# ===== 실행 =====
if __name__ == "__main__":
    symbol = input("📥 심볼 입력 (예: BTC): ").strip().upper()
    mode = input("🧠 모드 선택 (full / incremental): ").strip().lower()
    incremental = mode == "incremental"
    train_model(symbol, incremental=incremental)
