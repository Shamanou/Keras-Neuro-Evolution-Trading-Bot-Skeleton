class Wallet:
    def __init__(self, starting_cash, starting_price, trading_fee):
        self.starting_cash = starting_cash
        self.trading_fee = trading_fee
        self.cash_wallet = [starting_cash] * len(starting_price)
        self.btc_wallet = [0] * len(starting_price)
        self.is_holding = [False] * len(starting_price)
        self.starting_price = starting_price
        self.current_buy = [None] * len(starting_price)
        self.old_cash_wallet = [0] * len(starting_price)
        self.cash_history = []
        self.trade_history = []

    def buy(self, idx, price):
        if not self.is_holding[idx]:
            self.current_buy[idx] = price
            self.btc_wallet[idx] = self.cash_wallet[idx] / price * (1 - self.trading_fee)
            self.cash_history.append([idx, self.cash_wallet[idx]])
            self.trade_history.append(
                [idx, price, self.cash_wallet[idx], self.cash_wallet[idx] / price * (1 - self.trading_fee),
                 self.btc_wallet[idx]])

            self.old_cash_wallet[idx] = self.cash_wallet[idx]
            self.cash_wallet[idx] = 0
            self.is_holding[idx] = True

    def sell(self, idx, price):
        if self.is_holding[idx]:
            self.cash_wallet[idx] = self.btc_wallet[idx] * price * (1 - self.trading_fee)
            self.cash_history.append([idx, self.cash_wallet[idx]])
            self.trade_history.append(
                [idx, price, self.btc_wallet[idx], self.cash_wallet[idx] / self.old_cash_wallet[idx] * 100 - 100,
                 self.btc_wallet[idx]])

            self.btc_wallet[idx] = 0
            self.is_holding[idx] = False

    def get_holding_earnings(self, final_price):
        return (final_price / self.starting_price[-1]) * 100 - 100

    def get_swing_earnings(self, idx, final_price):
        self.sell(idx, final_price)
        return (self.cash_wallet[idx] / self.starting_cash) * 100 - 100

    def dump_trades(self, filename):
        f = open(filename, 'w')
        just = 26

        f.write("idx".ljust(just))
        f.write("price".ljust(just))
        f.write("cash_wallet".ljust(just))
        f.write("percent_change".ljust(just))
        f.write("btc_wallet".ljust(just))
        f.write("\n")

        for trade in self.trade_history:
            for el in trade:
                f.write("{:f}".format(el).ljust(just))
            f.write("\n")


if __name__ == '__main__':
    STARTING_CASH = 10
    STARTING_PRICE = [10]
    TRADING_FEE = 0.01

    WALLET = Wallet(STARTING_CASH, STARTING_PRICE, TRADING_FEE)

    WALLET.buy(0, 15)
    WALLET.sell(0, 20)

    print(WALLET.get_swing_earnings(0, -50))
