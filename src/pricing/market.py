class Market:
    def __init__(self, S0, r, sigma, D=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.D = D if D is not None else []