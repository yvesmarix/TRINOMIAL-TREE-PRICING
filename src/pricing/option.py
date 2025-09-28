class Option:
    def __init__(self, K, option_type, T):
        self.K = K
        self.option_type = option_type
        self.T = T

    @staticmethod
    def check_option_type(option_type):
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def payoff(self, S):
        if self.option_type == 'call':
            return max(S - self.K, 0)
        elif self.option_type == 'put':
            return max(self.K - S, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")