# model/CandleData.py

class CandleData:
    def __init__(self, timestamp, open_, high, low, close, volume):
        self.timestamp = timestamp  # Unix timestamp in ms
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume