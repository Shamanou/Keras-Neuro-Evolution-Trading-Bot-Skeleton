from time import time

import ccxt
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential

from utils.Population import Population

MARKETS = ccxt.kraken().load_markets()


def build_model():
    num_inputs = 73
    hidden_nodes = 100
    num_outputs = 3

    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=num_inputs))
    model.add(Dense(num_outputs, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')

    return model


if __name__ == '__main__':
    # suppress tf GPU logging
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    POP_SIZE = 100
    MUTATION_RATE = 0.05
    MUTATION_SCALE = 0.3
    STARTING_CASH = 2.5
    TRADING_FEE = 0.01
    GENERATIONS = 150

    # generate random test data
    np.random.seed(42)
    inputs = ccxt.kraken().fetch_tickers(MARKETS.keys())
    prices = [float(inputs[key]['info']['a'][0]) for key in inputs]
    inputs = np.random.rand(len(inputs), 2).tolist()
    inputs = [(int(i * len(inputs)), x * 10) for i, x in inputs]

    # build initial population
    pop = Population(POP_SIZE, build_model, MUTATION_RATE, MUTATION_SCALE, STARTING_CASH, TRADING_FEE)

    # run defined number of evolutions
    for i in range(GENERATIONS):
        start = time()
        pop.evolve(inputs, prices)
        print('\n\nDuration: {0:.2f}s'.format(time() - start))
