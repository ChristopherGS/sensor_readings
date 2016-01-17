from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import cifar10
from sklearn import cross_validation

import pandas as pd  
from random import random
import numpy as np

from wrangle import prep

training_data = prep()
y = training_data['state'].values
X = training_data.drop(['state', 'index'], axis=1)


#ACCEL_Z   GYRO_X    GYRO_Y   GYRO_Z  rolling_median_x  rolling_median_y  rolling_median_z  rolling_median_gx  rolling_median_gy  rolling_median_gz

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

x1 = X_train['ACCEL_X'].values
x2 = X_train['ACCEL_Y'].values
x3 = X_train['ACCEL_Z'].values

_x1 = X_test['ACCEL_X'].values
_x2 = X_test['ACCEL_Y'].values
_x3 = X_test['ACCEL_Z'].values

print X_train.shape
print y_train.shape


#======================================================

"""

model = Sequential()
model.add(Embedding(3, 16, input_length=3))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

X = np.column_stack([x1, x2, x3])
Y = np.column_stack([y_train])

model.fit(X, Y, batch_size=16, nb_epoch=5)

score, acc = model.evaluate(np.column_stack([_x1, _x2, _x3]), np.column_stack([y_test]),show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

classification = model.predict_classes(np.column_stack([_x1, _x2, _x3]), verbose=1)

print classification
"""


#=========================================================
"""
print('Build model 2...')
model2 = Sequential()
model2.add(Embedding(3, 128, input_length=100))
model2.add(LSTM(128))  # try using a GRU instead, for fun
model2.add(Dropout(0.5))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")

print("Train...")

X = np.column_stack([x1, x2, x3])
Y = np.column_stack([y_train])

_X = np.column_stack([_x1, _x2, _x3])
_Y = np.column_stack([y_test])

model2.fit(X, Y, batch_size=32, nb_epoch=5,
          validation_data=(_X, _Y), show_accuracy=True)
score, acc = model2.evaluate(_X, _Y,
                            batch_size=32,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

"""

#====================================================

"""
NOT SURE ABOUT TIMESTEPS

https://github.com/bigaidream/subsets_ml_cookbook/blob/d4e9e8def2068c83390257d0b5aed9072bf4ece6/dl/theano/theano_keras_sequence2sequence.md

n_in_out = 3
n_hidden = 100
n_samples = 2297
n_timesteps = 400

X = [len(X_train), n_timesteps, np.column_stack([x1, x2, x3])]
Y = np.column_stack([y_train])

_X = np.column_stack([_x1, _x2, _x3])
_Y = np.column_stack([y_test])

model3 = Sequential()

model3.add(GRU( n_hidden, input_dim = n_in_out, return_sequences=True))
model3.add(TimeDistributedDense(n_in_out, input_dim = n_hidden))
model3.compile(loss='mse', optimizer='rmsprop')

X_final = np.random.random((n_samples, n_timesteps, n_in))
Y_final = np.random.random((n_samples, n_timesteps, n_out))

model.fit(X, Y, nb_epoch=10, validation_data=(_X, _Y), show_accuracy=True)


"""

1) 
An optimizer is one of the two arguments required for compiling a Keras model:

# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd') #Stochastic gradient descent


2) 
An objective function (or loss function, or optimization score function) is one of the two parameters required to compile a model

3) Keras has two models: Sequential, a linear stack of layers, and Graph, a directed acyclic graph of layers.

4) Activations can either be used through an Activation layer, or through the activation argument supported by all forward layers:

from keras.layers.core import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
is equivalent to:

model.add(Dense(64, activation='tanh'))

5) Apply Dropout to the input. Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, 
which helps prevent overfitting.


6) Dense

keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', 
	weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, 
	b_constraint=None, input_dim=None)
Just your regular fully connected NN layer.


7) One epoch consists of one full training cycle on the training set. 
Once every sample in the set is seen, you start again - marking the beginning of the 2nd epoch.

8) To discuss: Should try GRU instead of LSTM
"""