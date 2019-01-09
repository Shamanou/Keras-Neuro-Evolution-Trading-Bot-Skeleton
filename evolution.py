from time import time

import ccxt
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential

from utils.Population import Population

markets = ccxt.kraken().load_markets()


def build_model():
    num_inputs = 2
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

    pop_size = 50
    mutation_rate = 0.05
    mutation_scale = 0.3
    starting_cash = 2.5
    trading_fee = 0.01
    generations = 150

    # generate random test data
    np.random.seed(42)
    inputs = ccxt.kraken().fetch_tickers(markets.keys())
    prices = [float(inputs[key]['info']['a'][0]) for key in inputs]
    inputs = np.random.rand(len(inputs),2).tolist()
    inputs = [ (int(i*len(inputs)),x*10) for i,x in inputs ]

    # build initial population
    pop = Population(pop_size, build_model, mutation_rate, mutation_scale, starting_cash, trading_fee)

    # run defined number of evolutions
    for i in range(generations):
        start = time()
        pop.evolve(inputs, prices)
        print('\n\nDuration: {0:.2f}s'.format(time() - start))
