# model/PriceInfo.py
class PriceInfo:
    def __init__(self, exchange: str, symbol: str, currency: str, price: float):
        self.exchange = exchange
        self.symbol = symbol
        self.currency = currency
        self.price = float(price)