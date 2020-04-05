class Pair:
    def __init__(self, symbol1: str, symbol2: str, equity_allocation: float, wide_spread: float = 1.1,
                 tight_spread: float = 0.9, trade_window: int = 5):
        self.sym1 = symbol1
        self.sym2 = symbol2
        self.equity = equity_allocation
        self.ws = wide_spread
        self.ts = tight_spread
        self.trade_window = trade_window
