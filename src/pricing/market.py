class Market:
    def __init__(self, S0, r, sigma, dividend = None, dividend_date = None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.dividend = dividend
        self.dividend_date = dividend_date