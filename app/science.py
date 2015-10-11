# app/science.py

import os
import pandas as pd
import numpy as np
import random
import math

from sklearn import svm

"""
Support vector machines (SVMs) are a set of supervised learning methods 
used for classification, regression and outliers detection.
"""

_basedir = os.path.abspath(os.path.dirname(__file__))
UPLOADS = 'api/uploads'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)


#dated
#filename = 'badboy_combined.csv'
#unlabelled_filename = 'badboy_unlabelled2.csv'

#PLAY_DATA = UPLOAD_FOLDER + '/' + filename
#PLAY_DATA2 = UPLOAD_FOLDER + '/' + unlabelled_filename

#df = pd.read_csv(PLAY_DATA)
#df_unlabelled = pd.read_csv(PLAY_DATA2)

#print df


def sql_to_pandas():
    pass

def pandas_cleanup(df):
    columns = []
    df_clean = df[['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'timestamp', 'experiment_id']]
    return df_clean


#class HMM:
#   def __init__(self):
#       pass

def my_hmm():

    """
    Probability of being in a particular state at step i is known once we know
    what state we were in at step i-1

    Probability of seeing a particular emission
    at step i is known once we know what state we were in at step i.

    a sequence of observable X variable is generated 
    by a sequence of internal hidden state Z.
    """

    # linear dynamical system model needs to do the heavy lifting --> which can model the dynamics of a punch

    # the HMM will act as a switching mechanism, so that each hidden state will represent a gesture

    # but the gesture is learned by the linear dynamical system 

    # Question is how to connect the two together

    # HACK - increase the components of the HMM...not sure

    states = ('punch', 'other')

    observations = ('accelerometer_x', 'accelerometer_y', 'accelerometer_z')

    start_probability = {'punch': 0.4, 'other': 0.6}

    transition_probability = {
        'punch' : {'punch': 0.9, 'other': 0.1},
        'other' : {'punch': 0.2, 'other': 0.8}
    }

    # ARBITRARY VALUES - need to train the model
    emission_probability = {
        'punch' : {'accelerometer_x': 0.5, 'accelerometer_y': 0.4, 'accelerometer_z': 0.1},
        'other' : {'accelerometer_x': 0.1, 'accelerometer_y': 0.3, 'accelerometer_z': 0.6}
    }

    # QUESTION HOW DOES THIS WORK FOR CONTINUOUS SEQUENCES?

    startprob = np.array([0.4, 0.6])
    transition_matrix = np.array([[0.9, 0.1],
                                [0.2, 0.8]])

    # need to understand the uncertainty of the observations
    # by understanding which state are they from
    # analysis of variance

    # so the mean is the values of the clusters

    # covariance (variance between two different random variables) 

    """
    The covariance of each component
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))

    creates array of 4 2x2 matrices with 0.5 on the diagonals and zero on the off diagonals

    4 links to the number of components

    how to check means? and covars? 

    """

    # means --> (array, shape (n_components, n_features)) Mean parameters for each state.
    # in this case 2 x 3
    # means = ?
    # covars = ?
    # model = hmm.GuassianHMM(3, "full", startprob, transition_matrix)
    # model.means = means
    # model.covars = covars
    # X, Z = model.sample(100) The observable vs. hidden state probabilities


    # --------------------------------------------------
    # TRAIN MODEL

    """
    List of array-like observation sequences, 
    each of which has shape (n_i, n_features), where n_i is the length of the i_th observation.
    """

    # COULD IT BE THAT I HAVE MISUNDERSTOOD THE TRAINING HERE?

    x = df['accelerometer_x'].values
    y = df['accelerometer_y'].values
    z = df['accelerometer_z'].values

    X = np.column_stack([x, y, z])
    print X
    print X.shape

    thurs_model = Model( name="Punch-Other" )
    # Emission probabilities

    # looks for discrete distribution
    punch = State( DiscreteDistribution({ 'walk': 0.1, 'shop': 0.4, 'clean': 0.5 }) )
    other = State( DiscreteDistribution({ 'walk': 0.6, 'shop': 0.3, 'clean': 0.1 }) )



    ###############################################################################
    # Run Gaussian HMM
    print("fitting to HMM and decoding ...")

    # n_components : Number of states.
    #_covariance_type : string describing the type of covariance parameters to use. 
    # Must be one of 'spherical', 'tied', 'diag', 'full'. Defaults to 'diag'.
    model = GaussianHMM(n_components=2, covariance_type="diag").fit(X)

    # predict the optimal sequence of internal hidden state

    # Get the unlabelled data
    _x = df_unlabelled['accelerometer_x'].values
    _y = df_unlabelled['accelerometer_y'].values
    _z = df_unlabelled['accelerometer_y'].values

    Z = np.column_stack([_x, _y, _z])

    # print Z
    print Z.shape

    hidden_states = model.predict(Z)

    print("done\n")

    ####################################
    
    print("means and vars of each hidden state")
    for i in range(model.n_components):
        print("%dth hidden state" % i)
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()




    print hidden_states[1:50]

    # 1 = punch
    # 0 = other


#my_hmm()






