import random

import ccxt
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import model_from_json

from utils.Agent import Agent

MARKETS = ccxt.kraken().load_markets()


class Population(object):
    def __init__(self, pop_size, model_builder, mutation_rate, mutation_scale, starting_cash,
                 trading_fee, big_bang=True, remove_sleep=False, one_output=False, dump_trades=False, dump_file=None):
        self.pop_size = pop_size
        self.agents = []

        self.model_builder = model_builder
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.starting_cash = starting_cash
        inputs = ccxt.kraken().fetch_tickers(MARKETS.keys())
        self.starting_price = [float(inputs[key]['info']['a'][0]) for key in inputs]
        self.trading_fee = trading_fee

        self.has_sleep_functionality = not remove_sleep
        self.has_one_output = one_output

        self.generation_number = 1
        self.output_width = 5

        self.dump_trades = dump_trades
        self.dump_file = dump_file

        plt.ion()

        if big_bang:
            for i in range(self.pop_size):
                # print("\rbuilding agents {:.2f}%...".format((i + 1) / self.pop_size * 100))
                agent = Agent(self, i)
                self.agents.append(agent)
        print("")

    def set_preexisting_agent_base(self, model):
        self.agents = []
        for i in range(self.pop_size):
            self.agents.append(Agent(self, i, inherited_model=model))

    def validate(self, inputs_list, prices_list, output_width=5, plot_best=False):
        season_num = 10
        self.batch_feed_inputs(inputs_list, prices_list)
        self.print_profits(output_width, prices_list)
        self.normalize_fitness()
        self.sort_by_decreasing_fitness()
        if plot_best:
            self.plot_best_agent(prices_list, season_num)
        if self.dump_trades:
            self.agents[0].wallet.dump_trades(self.dump_file)

    def evolve(self, inputs_list, prices_list, output_width=15, plot_best=False, season_num=None):
        print("\n======================\ngeneration number {}\n======================".format(self.generation_number))

        self.batch_feed_inputs(inputs_list, prices_list)
        self.print_profits(output_width, prices_list)
        self.normalize_fitness()
        self.sort_by_decreasing_fitness()
        if plot_best:
            self.plot_best_agent(prices_list, season_num)
        if self.dump_trades:
            self.agents[0].wallet.dump_trades(self.dump_file)

        self.save_best_agent()
        self.generate_next_generation()

    def evolve_over_seasons(self, inputs_list, prices_list, num_seasons=4, epochs_per_season=1, roulette=False):
        inputs_seasons = np.array_split(inputs_list, num_seasons)
        prices_seasons = np.array_split(prices_list, num_seasons)

        if not roulette:
            cnt = 0
            for inputs, prices in zip(inputs_seasons, prices_seasons):
                for i in range(epochs_per_season):
                    self.evolve(inputs, prices, plot_best=True, season_num=cnt + 1)
                cnt += 1
        else:
            l = range(num_seasons)
            for j in range(num_seasons):
                choice = random.choice(l)
                inputs = inputs_seasons[choice]
                prices = prices_seasons[choice]

                for i in range(epochs_per_season):
                    self.evolve(inputs, prices, plot_best=True, season_num=choice + 1)

    def batch_feed_inputs(self, inputs_list, prices_list):
        for i in range(len(self.agents)):
            self.agents[i].batch_act(inputs_list, prices_list)
            # print("\rFeeding inputs {:.2f}%...".format((i + 1) / self.pop_size * 100))

    def normalize_fitness(self):
        print("normalizing fitness...")

        scores_arr = []
        for agent in self.agents:
            scores_arr.append(agent.score)

        mi = min(scores_arr)
        ma = max(scores_arr)
        den = ma - mi

        for i in range(len(self.agents)):
            try:
                new_score = ((self.agents[i].score - mi) / den) ** 2
            except:
                new_score = (self.agents[i].score - mi) ** 2
            self.agents[i].score = new_score

        s = 0
        for agent in self.agents:
            s += agent.score

        for i in range(self.pop_size):
            if s != 0:
                self.agents[i].fitness = self.agents[i].score / s
            else:
                self.agents[i].fitness = 0

    def pool_selection(self):
        idx, cnt = 0, 0
        r = np.random.random()

        while cnt < r and idx != len(self.agents):
            cnt += self.agents[idx].fitness
            idx += 1

        return idx - 1

    def generate_next_generation(self):
        # find models to mutate to next generation
        new_agents_idx = []
        for i in range(self.pop_size):
            new_agents_idx.append(self.pool_selection())

        # temporarily save models and clear session
        configs, weights = [], []
        for model_idx in new_agents_idx:
            configs.append(self.agents[model_idx].model.to_json())
            weights.append(self.agents[model_idx].model.get_weights())

        K.clear_session()

        # reload models
        new_agents = []
        for i in range(self.pop_size):
            # print("\rcreating next generation {0:.2f}%...".format((i + 1) / self.pop_size * 100))
            loaded = model_from_json(configs[i])
            loaded.set_weights(weights[i])
            new_agents.append(Agent(self, i, inherited_model=loaded))
        print("")

        self.agents = new_agents
        self.generation_number += 1

        # mutation scale decay
        # self.mutation_scale *= 0.95

    def sort_by_decreasing_fitness(self):
        self.agents.sort(key=lambda x: x.fitness, reverse=True)

    def print_profits(self, output_width, prices):
        c = 0
        profit_arr = []
        for agent in self.agents:
            profit_tmp = []
            for i in range(len(prices)):
                profit_tmp.append(agent.wallet.get_swing_earnings(i, prices[i]))
                profit_tmp.sort()
            profit_arr.append(profit_tmp)
        profit_arr = np.reshape(profit_arr, (len(profit_arr[0]), len(profit_arr)))
        profit_arr.sort()

        output_str = "\nmaximum profit: {0:.5f}%\n".format(np.average(profit_arr))
        # for score in profit_arr:
        #     output_str += "{0:.5f}%".format(max(score)).ljust(12)
        #     c += 1
        #     if c % output_width == 0:
        #         output_str += "\n"
        print(output_str)

    def save_best_agent(self):
        self.sort_by_decreasing_fitness()
        self.agents[0].save("saved_agent/best_agent")

    def plot_best_agent(self, prices, season_num=None):
        wallet_values = []
        if self.agents[0].wallet.cash_history:
            for i, hist in enumerate(self.agents[0].wallet.cash_history):
                wallet_values.append([i, hist[0], hist[1]])

            plt.clf()

            plt.figure(1)
            if season_num != None:
                plt.suptitle("Trading Bot Generation {} Season {}".format(self.generation_number, season_num))
            else:
                plt.suptitle("Trading Bot Generation {}".format(self.generation_number))

            ax1 = plt.subplot(211)
            ax1.set_ylabel("Price Graph")
            ax1.plot(prices)

            wallet_values = sorted(wallet_values, key=lambda x: x[1])

            tmp = wallet_values[0][1]
            tmp_arr = []
            arr = []
            for i, x, y in wallet_values:
                if tmp == x:
                    tmp_arr.append(y)
                else:
                    arr.append(tmp_arr)
                    tmp_arr = []
                tmp = x

            ax2 = plt.subplot(212)
            ax2.set_ylabel("Cash Wallet Value")
            for el in arr:
                ax2.plot(range(len(el)), el)

            # plt.show()
            plt.draw()
            plt.pause(0.001)
