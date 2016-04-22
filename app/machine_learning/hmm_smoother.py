import numpy as np
import pandas as pd
import os
from hmmlearn import hmm
from utilities import convert_to_words

n_components = 8 # ('ybc', 'ymount', 'ysc', 'ycg', 'ocg', 'osc_mount', 'obc', 'other')
startprob = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65,]) # users will probably turn on sensor standing

"""
probability of these positions given current state:

your_mount' if v == 0 
else 'your_side_control' if v == 1
else 'your_closed_guard' if v == 2
else 'your_back_control' if v == 3
else 'opponent_mount_or_sc' if v == 4
else 'opponent_closed_guard' if v == 5
else 'opponent_back_control' if v == 6
else 'OTHER' if v == 7

transition_probability = {
        'ymt' : {'ymount': 0.800, 'ysc': 0.050, 'ycg': 0.010, 'ybc': 0.050, 'osc_mount': 0.001, 'ocg': 0.050, 'obc': 0.001, 'other': 0.038},
        'ysc' : {'ymount': 0.100, 'ysc': 0.800, 'ycg': 0.010, 'ybc': 0.010, 'osc_mount': 0.001, 'ocg': 0.050, 'obc': 0.001, 'other': 0.028},
        'ycg' : {'ymount': 0.010, 'ysc': 0.050, 'ycg': 0.800, 'ybc': 0.010, 'osc_mount': 0.050, 'ocg': 0.001, 'obc': 0.001, 'other': 0.078},
        'ybc' : {'ymount': 0.050, 'ysc': 0.010, 'ycg': 0.050, 'ybc': 0.800, 'osc_mount': 0.001, 'ocg': 0.010, 'obc': 0.001, 'other': 0.078},
        'omt' : {'ymount': 0.001, 'ysc': 0.050, 'ycg': 0.010, 'ybc': 0.050, 'osc_mount': 0.800, 'ocg': 0.050, 'obc': 0.001, 'other': 0.038},
        'ocg' : {'ymount': 0.100, 'ysc': 0.050, 'ycg': 0.010, 'ybc': 0.010, 'osc_mount': 0.001, 'ocg': 0.800, 'obc': 0.001, 'other': 0.028},
        'obc' : {'ymount': 0.010, 'ysc': 0.050, 'ycg': 0.001, 'ybc': 0.010, 'osc_mount': 0.050, 'ocg': 0.001, 'obc': 0.800, 'other': 0.078},
        'oth' : {'ymount': 0.050, 'ysc': 0.010, 'ycg': 0.050, 'ybc': 0.078, 'osc_mount': 0.001, 'ocg': 0.010, 'obc': 0.001, 'other': 0.800}
     }
"""

transmat = np.array([
                    [0.800, 0.050, 0.010, 0.050, 0.001, 0.050, 0.001, 0.038], 
                    [0.100, 0.800, 0.010, 0.010, 0.001, 0.050, 0.001, 0.028], 
                    [0.010, 0.050, 0.800, 0.010, 0.050, 0.001, 0.001, 0.078], 
                    [0.050, 0.010, 0.050, 0.800, 0.001, 0.010, 0.001, 0.078],
                    [0.001, 0.050, 0.010, 0.050, 0.800, 0.050, 0.001, 0.038],
                    [0.100, 0.050, 0.010, 0.010, 0.001, 0.800, 0.001, 0.028],
                    [0.010, 0.050, 0.001, 0.010, 0.050, 0.001, 0.800, 0.078],
                    [0.050, 0.010, 0.050, 0.078, 0.001, 0.010, 0.001, 0.800],
                    ])

"""
probability of these positions given current state:

your_mount' if v == 0 
else 'your_side_control' if v == 1
else 'your_closed_guard' if v == 2
else 'your_back_control' if v == 3
else 'opponent_mount_or_sc' if v == 4
else 'opponent_closed_guard' if v == 5
else 'opponent_back_control' if v == 6
else 'OTHER' if v == 7

emission_probability = {
        'ymt' : {'ymount': 0.600, 'ysc': 0.050, 'ycg': 0.010, 'ybc': 0.050, 'osc_mount': 0.001, 'ocg': 0.250, 'obc': 0.001, 'other': 0.038},
        'ysc' : {'ymount': 0.100, 'ysc': 0.800, 'ycg': 0.010, 'ybc': 0.010, 'osc_mount': 0.001, 'ocg': 0.050, 'obc': 0.001, 'other': 0.028},
        'ycg' : {'ymount': 0.010, 'ysc': 0.050, 'ycg': 0.600, 'ybc': 0.010, 'osc_mount': 0.250, 'ocg': 0.001, 'obc': 0.001, 'other': 0.078},
        'ybc' : {'ymount': 0.050, 'ysc': 0.010, 'ycg': 0.050, 'ybc': 0.700, 'osc_mount': 0.001, 'ocg': 0.010, 'obc': 0.101, 'other': 0.078},
        'omt' : {'ymount': 0.001, 'ysc': 0.050, 'ycg': 0.210, 'ybc': 0.050, 'osc_mount': 0.600, 'ocg': 0.050, 'obc': 0.001, 'other': 0.038},
        'ocg' : {'ymount': 0.300, 'ysc': 0.050, 'ycg': 0.010, 'ybc': 0.010, 'osc_mount': 0.001, 'ocg': 0.500, 'obc': 0.001, 'other': 0.028},
        'obc' : {'ymount': 0.010, 'ysc': 0.050, 'ycg': 0.001, 'ybc': 0.110, 'osc_mount': 0.050, 'ocg': 0.001, 'obc': 0.700, 'other': 0.078},
        'oth' : {'ymount': 0.050, 'ysc': 0.010, 'ycg': 0.050, 'ybc': 0.078, 'osc_mount': 0.001, 'ocg': 0.010, 'obc': 0.001, 'other': 0.800}
     }
"""

emissionprob = np.array([
                        [0.600, 0.050, 0.010, 0.050, 0.001, 0.250, 0.001, 0.038], 
                        [0.100, 0.800, 0.010, 0.010, 0.001, 0.050, 0.001, 0.028], 
                        [0.010, 0.050, 0.600, 0.010, 0.250, 0.001, 0.001, 0.078], 
                        [0.050, 0.010, 0.050, 0.700, 0.001, 0.010, 0.101, 0.078],
                        [0.001, 0.050, 0.210, 0.050, 0.600, 0.050, 0.001, 0.038],
                        [0.300, 0.050, 0.010, 0.010, 0.001, 0.500, 0.001, 0.028],
                        [0.010, 0.050, 0.001, 0.110, 0.050, 0.001, 0.700, 0.078],
                        [0.050, 0.010, 0.050, 0.078, 0.001, 0.010, 0.001, 0.800],
                        ])

# Hidden Markov Model with multinomial (discrete) emissions
model = hmm.MultinomialHMM(n_components=n_components,
                           n_iter=10,
                           verbose=False)

model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob


def apply_hmm(data):
	"""smooth the predictions from the Random Forest classifier
	using HMM logic"""

	data_ = np.array(data)
	n_samples = len(data)
	raw_data = data_.reshape((n_samples, -1))
	result = model.decode(raw_data, algorithm='viterbi')
	result_words = convert_to_words(result[1])
	print 'result words: {}'.format(result_words)
	print 'result accuracy: {}'.format(result[0])

	return result[1]

