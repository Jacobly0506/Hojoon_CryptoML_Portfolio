# 💹 CryptoTradingProject

A hybrid cryptocurrency trading system that uses **real-time market data**, **machine learning models**, and **technical indicators** to make trading decisions.  
Originally implemented with JavaScript (TensorFlow.js), the project was **entirely migrated to Python** for better performance, flexibility, and maintainability.

---

## 🔄 Why I Transitioned from JS to Python

This project began with TensorFlow.js to experiment with client-side machine learning. However, I made a strategic switch to Python for the following reasons:

- **Broader ecosystem**: Libraries like `TensorFlow`, `Keras`, `pandas`, and `NumPy` are more mature and well-documented in Python.
- **Better ML support**: Python offers robust ML tooling (training, saving, evaluation, preprocessing).
- **System-level integration**: Easier interaction with OS, data pipelines, and REST APIs.
- **Scalability**: Python can support multiprocessing, memory control, and backend integration more effectively.

---

## 📁 Project Structure Overview

\`\`\`
CryptoTradingProject/
├── main.py                    # CLI-based interface with full menu
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git tracking rules
│
├── ml/                        # ML model training & prediction scripts
│   ├── data/                  # (Optional) Historical price data
│   ├── models/                # Saved .keras models per timeframe
│   ├── train_15m.py           # Train model with 15m OHLCV
│   ├── train_1h.py            # Train model with 1h OHLCV
│   ├── train_4h.py            # Train model with 4h OHLCV
│   ├── train_1d.py            # Train model with 1d OHLCV
│   ├── predict_*.py           # Predict using trained models
│   ├── train_summary.py       # Multiprocessing batch trainer
│   └── predict_summary.py     # Unified prediction across timeframes
│
├── model/
│   ├── CandleData.py          # Custom dataclass for OHLCV candles
│   └── PriceInfo.py           # Result holder for prediction + indicators
│
├── services/
│   └── BinanceService.py      # API wrapper for Binance US REST endpoints
│
├── utils/
│   └── Indicators.py          # Technical indicator calculators (SMA, RSI, VWAP, etc.)
\`\`\`

---

## ⚙️ Features

- 🔁 **Real-time Data Fetching**
  - Binance US API integration
  - Fetch OHLCV and volume data for any symbol

- 📊 **Technical Indicators**
  - SMA (Simple Moving Average)
  - EMA, RSI, VWAP, Bollinger Bands
  - ATR (Average True Range), OBV (On-Balance Volume)

- 🧠 **Deep Learning Predictions**
  - Trained models for multiple timeframes (15m, 1h, 4h, 1d)
  - Multivariate LSTM-style prediction
  - Multi-output forecast: Close, High, Low

- 🧮 **Hybrid Decision Engine**
  - Combines technical indicators + ML predictions
  - Returns a 7-tier actionable signal:
    - 적극 매수 (Strong Buy) → 적극 매도 (Strong Sell)

- ⚡ **Multiprocessing Training Pipeline**
  - Parallel model training across timeframes

---

## 🧪 How to Run

### 1. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Launch Main CLI Menu

\`\`\`bash
python main.py
\`\`\`

### 3. Available Options in CLI Menu

\`\`\`
1. Current Analysis + ML Prediction
2. Model-train
3. Model-view
4. Model-clear
5. Ensemble-train
6. Ensemble-predict
7. Ensemble-Model-clear
8. Hybrid Decision Engine ✅
\`\`\`

---

## 📌 Key Design Choices

- **Modularized structure**: Easy to expand with new indicators or timeframes
- **Timeframe-specific models**: Allows short-term vs long-term strategic analysis
- **Full offline capability**: Models can be used without continuous API calls once trained

---

## 🌱 Future Enhancements

- [ ] Real-time trading simulation or backtest mode
- [ ] Web dashboard for live visualization
- [ ] Model performance monitoring (MSE, R²)
- [ ] Telegram / Discord alerts
- [ ] Integration with broker API for live execution

---

## 📜 License

This project is open-source and uses the [MIT License](LICENSE).

---

## 🙋 About the Author

Developed by **Hojoon Lee**, a graduate student at Carnegie Mellon University  
Passionate about data-driven finance, algorithmic trading, and hybrid AI systems.


---

## 📚 References

This project was supported in part by AI-assisted tools such as **OpenAI ChatGPT**, which contributed to:

- Code architecture suggestions
- Modular structure planning
- CLI design for Python
- Markdown documentation and README formatting

All implementation, training, and customization were developed by **Hojoon Lee**.  
This repository serves as a part of my professional **portfolio**, showcasing full-stack ML integration for crypto finance.

