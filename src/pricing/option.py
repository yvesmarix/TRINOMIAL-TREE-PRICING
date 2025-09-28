class Option:
    def __init__(self, K, option_type, T, option_class):
        self.K = K
        self.option_type = self.check_option_type(option_type)
        self.T = T
        self.option_class = self.check_option_class(option_class)

    @staticmethod
    def check_option_type(option_type):
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        return option_type.lower()

    @staticmethod
    def check_option_class(option_class):
        if option_class.lower() not in ['european', 'american']:
            raise ValueError("Invalid option type. Use 'european' or 'american'.")
        return option_class.lower()

    def payoff(self, S):
        if self.option_type == 'call':
            return max(S - self.K, 0)
        elif self.option_type == 'put':
            return max(self.K - S, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")