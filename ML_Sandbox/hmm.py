import numpy as np
from hmmlearn import hmm

np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")

model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                             [0.3, 0.5, 0.2],
                             [0.3, 0.3, 0.4]])
model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))
X, Z = model.sample(100)



states = ('ybc', 'ymount', 'ysc', 'ycg', 'ocg', 'osc_mount', 'obc', 'other')

observations = ('_ybc', '_ymount', '_ysc', '_ycg', '_ocg', '_osc_mount', '_obc', '_other')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}


# probability of these positions given current state
transition_probability = {
     'ybc' : {'ybc': 0.8, 'ymount': 0.05, 'ysc': 0.01, 'ycg': 0.05, 'ocg': 0.01, 'osc_mount': 0.001, 'obc': 0.001, 'other': 0.078},
     'ymount' : {'ybc': 0.05, 'ymount': 0.8, 'ysc': 0.05, 'ycg': 0.01, 'ocg': 0.05, 'osc_mount': 0.001, 'obc': 0.001, 'other': 0.048},
     'ysc' : {'ybc': 0.01, 'ymount': 0.1, 'ysc': 0.8, 'ycg': 0.01, 'ocg': 0.05, 'osc_mount': 0.001, 'obc': 0.001, 'other': 0.028},
     'ycg' : {'ybc': 0.01, 'ymount': 0.01, 'ysc': 0.05, 'ycg': 0.8, 'ocg': 0.001, 'osc_mount': 0.05, 'obc': 0.001, 'other': 0.078},
     'ocg' : {'ybc': 0.001, 'ymount': 0.01, 'ysc': 0.05, 'ycg': 0.001, 'ocg': 0.8, 'osc_mount': 0.05, 'obc': 0.01, 'other': 0.078},
     }



# The emission_probability represents how likely the patient is to feel on each day. 
# If he is healthy, there is a 50% chance that he feels normal; if he has a fever, there is a 60% chance that he feels dizzy.
emission_probability = {
    'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
    }