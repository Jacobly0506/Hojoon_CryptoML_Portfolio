import requests
from model.PriceInfo import PriceInfo
from model.CandleData import CandleData

class BinanceService:
    BASE_URL = "https://api.binance.us/api/v3"

    def fetch_price(self, input_symbol: str) -> PriceInfo:
        symbol = input_symbol.upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        endpoint = f"{self.BASE_URL}/ticker/price?symbol={symbol}"
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        price = data["price"]
        base_symbol = symbol.replace("USDT", "")
        return PriceInfo("Binance", base_symbol, "USD", price)

    def fetch_historical_prices(self, symbol: str, limit: int, interval: str) -> list[float]:
        if not interval:
            raise ValueError("Timeframe (interval) must be explicitly provided.")
        symbol = symbol.upper() + "USDT"
        endpoint = f"{self.BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(endpoint)
        response.raise_for_status()
        klines = response.json()
        return [float(kline[4]) for kline in klines]

    def fetch_historical_price_volume(self, symbol: str, limit: int, interval: str) -> list[list[float]]:
        if not interval:
            raise ValueError("Timeframe (interval) must be explicitly provided.")
        symbol = symbol.upper() + "USDT"
        endpoint = f"{self.BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(endpoint)
        response.raise_for_status()
        klines = response.json()
        return [[float(k[4]), float(k[5])] for k in klines]

    def fetch_historical_candle_data(self, symbol: str, limit: int, interval: str) -> list[CandleData]:
        if not interval:
            raise ValueError("Timeframe (interval) must be explicitly provided.")
        symbol = symbol.upper() + "USDT"
        endpoint = f"{self.BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(endpoint)
        response.raise_for_status()
        klines = response.json()

        candles = []
        for k in klines:
            timestamp = int(k[0])  # <--- ✅ open_time (Unix ms)
            open_ = float(k[1])
            high = float(k[2])
            low = float(k[3])
            close = float(k[4])
            volume = float(k[5])
            candles.append(CandleData(timestamp, open_, high, low, close, volume))
        return candles
