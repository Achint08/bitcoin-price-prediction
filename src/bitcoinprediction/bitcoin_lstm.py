import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
import gc

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from preprocess_data import *
from model import *
from plot_data import *

neurons = 512                 # number of hidden units in the LSTM layer


btc_data = get_market_data("bitcoin", tag='BTC')
eth_data = get_market_data("ethereum", tag='ETH')

show_plot(btc_data, tag='BTC')
show_plot(eth_data, tag='ETH')

market_data = merge_data(btc_data, eth_data)
model_data = create_model_data(market_data)
train_set, test_set = split_data(model_data)

model_data.head()

train_set = train_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

X_train = create_inputs(train_set)
Y_train_btc = create_outputs(train_set, coin='BTC')
X_test = create_inputs(test_set)
Y_test_btc = create_outputs(test_set, coin='BTC')

Y_train_eth = create_outputs(train_set, coin='ETH')
Y_test_eth = create_outputs(test_set, coin='ETH')

X_train, X_test = to_array(X_train), to_array(X_test)

print (np.shape(X_train), np.shape(X_test), np.shape(Y_train_btc), np.shape(Y_test_btc))
print (np.shape(X_train), np.shape(X_test), np.shape(Y_train_eth), np.shape(Y_test_eth))

# clean up the memory
gc.collect()

# random seed for reproducibility
np.random.seed(202)

# initialise model architecture
btc_model = build_model(X_train, output_size=1, neurons=neurons)

# train model on data
btc_history = btc_model.fit(X_train, Y_train_btc, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test_btc), shuffle=False)

plot_results(btc_history, btc_model, Y_train_btc, coin='BTC')

## for ethnium coin
# # clean up the memory
# gc.collect()

# # random seed for reproducibility
# np.random.seed(202)

# # initialise model architecture
# eth_model = build_model(X_train, output_size=1, neurons=neurons)

# # train model on data
# eth_history = eth_model.fit(X_train, Y_train_eth, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test_eth), shuffle=False)

# plot_results(eth_history, eth_model, Y_train_eth, coin='ETH')b


