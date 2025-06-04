# CryptoTradingProject

A hybrid cryptocurrency trading system leveraging **real-time market data**, **advanced machine learning**, and **technical indicators** to make data-driven trading decisions.
Originally built with JavaScript (TensorFlow.js), the project was fully migrated to Python for improved performance, extensibility, and reliability.

---

## Why I Transitioned from JS to Python

This project began with TensorFlow.js to experiment with client-side machine learning. However, I made a strategic switch to Python for the following reasons:

- **Broader ecosystem**: Libraries like TensorFlow, Keras, pandas, and NumPy are more mature, well-supported, and production-ready in Python.
- **Better ML support**: Python offers robust ML tooling (training, saving, evaluation, preprocessing).
- **System-level integration**: Easier interaction with OS, data pipelines, and REST APIs.
- **Scalability**: Python can support multiprocessing, memory control, and backend integration more effectively.

---

## Project Structure Overview

CryptoTradingProject/<br>
├── main.py                    # CLI-based interface with full menu<br>
├── requirements.txt           # Python package dependencies<br>
├── .gitignore                 # Git tracking rules<br>
│<br>
├── ml/                        # ML model training & prediction scripts<br>
│   ├── data/                  # (Optional) Historical price data<br>
│   ├── models/                # Saved .keras models per timeframe<br>
│   ├── train_15m.py           # Train model with 15m OHLCV<br>
│   ├── train_1h.py            # Train model with 1h OHLCV<br>
│   ├── train_4h.py            # Train model with 4h OHLCV<br>
│   ├── train_1d.py            # Train model with 1d OHLCV<br>
│   ├── predict_*.py           # Predict using trained models<br>
│   ├── train_summary.py       # Multiprocessing batch trainer<br>
│   └── predict_summary.py     # Unified prediction across timeframes<br>
│<br>
├── model/<br>
│   ├── CandleData.py          # Custom dataclass for OHLCV candles<br>
│   └── PriceInfo.py           # Result holder for prediction + indicators<br>
│<br>
├── services/<br>
│   └── BinanceService.py      # API wrapper for Binance US REST endpoints<br>
│<br>
├── utils/<br>
│   └── Indicators.py          # Technical indicator calculators (SMA, RSI, VWAP, etc.)<br>

---

## Features

- **Real-time Data Fetching**
  - Binance US API integration
  - Fetch OHLCV and volume data for any symbol

- **Technical Indicators**
  - SMA (Simple Moving Average)
  - EMA, RSI, VWAP, Bollinger Bands
  - ATR (Average True Range), OBV (On-Balance Volume)

- **Deep Learning Predictions**
  - Trained models for multiple timeframes (15m, 1h, 4h, 1d)
  - Multivariate LSTM-style prediction
  - Multi-output forecast: Close, High, Low

- **Hybrid Decision Engine**
  - Combines technical indicators + ML predictions
  - Returns a 7-tier actionable signal:
    - 적극 매수 (Strong Buy) → 적극 매도 (Strong Sell)

- **Multiprocessing Training Pipeline**
  - Parallel model training across timeframes

---

## How to Run

**1. Install Dependencies**

   pip install -r requirements.txt

**2. Launch the CLI**
  
   python main.py

**3. Menu Options**

- Real-time Hybrid Analysis
- Model Training
- Price Prediction
- (and more...)

---

## Key Design Choices

- **Large Data Window**: Fetches and analyzes up to 1000 candles per symbol from the Binance API, enabling deep context for both ML models and technical indicators.
- **Flexible Timeframe Selection**: Easily switch between 1m, 15m, 1h, 4h, and 1d intervals for both training and prediction.
- **Modular Design**: All components—data fetching, feature engineering, model training, and prediction—are fully modular for rapid experimentation.

---

## Future Enhancements

- [ ] News Sentiment & Keywords Algorithm: Integrate a real-time news crawler and NLP-based keyword extraction to gauge market sentiment and automatically reflect critical news into trading signals.
- [ ] OHLC ML Prediction Ensemble: Ensemble multiple ML models for OHLC (Open, High, Low, Close) price prediction across different timeframes (15m, 1h, 4h, 1d) to increase robustness and accuracy.
- [ ] 1 - Minute (1m) Model Support:
    - Implement train_1m.py for ultra-short-term trading strategies.
    - Enable fast predictions (predict_1m.py) and automatic CSV saving for each run.
    - Support incremental saving to grow datasets as the system runs.
- [ ] And So On...

---

## License

This project is open-source and uses the [MIT License](LICENSE).

---

## About the Author

Developed by **Hojoon Lee**, a graduate student at Carnegie Mellon University  
Passionate about data-driven finance, algorithmic trading, and hybrid AI systems.


---

## References

This project was supported in part by AI-assisted tools such as **OpenAI ChatGPT**, which contributed to:

- Code architecture suggestions
- Modular structure planning
- CLI design for Python
- Markdown documentation and README formatting

All implementation, training, and customization were developed by **Hojoon Lee**.  
This repository serves as a part of my **portfolio**, showcasing full-stack ML integration for crypto finance.
