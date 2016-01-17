from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
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

model = Sequential()
model.add(Embedding(3, 16, input_length=3))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

X = np.column_stack([x1, x2, x3])
Y = np.column_stack([y_train])

model.fit(X, Y, batch_size=16, nb_epoch=10)
print model.predict(np.column_stack([_x1, _x2, _x3]))
print np.stack(y_test)


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
"""