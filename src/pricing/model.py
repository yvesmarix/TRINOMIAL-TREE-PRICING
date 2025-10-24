import datetime


class Model:

    def __init__(self, pricingDate: datetime.datetime):
        self.pricing_date: datetime.date = pricingDate

    @staticmethod
    def check_probability(probability: float, comment: str = ""):
        assert (0 <= probability <= 1), "Error: probability " + comment + " = " + Model.str_pc(probability) + "!"

    @staticmethod
    def str_pc(number: float, precision: int = 2) -> str:
        return ("{:." + str(int(precision)) + "%}").format(number)
