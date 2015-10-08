# app/science.py

import os
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pylab as pl
from yahmm import *
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

filename = 'badboy_combined.csv'
unlabelled_filename = 'badboy_unlabelled2.csv'

PLAY_DATA = UPLOAD_FOLDER + '/' + filename
PLAY_DATA2 = UPLOAD_FOLDER + '/' + unlabelled_filename

df = pd.read_csv(PLAY_DATA)
df_unlabelled = pd.read_csv(PLAY_DATA2)

print df


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


my_hmm()



#import numpy as np
#from copy import copy
#import matplotlib.pyplot as plt

"""
We will also use A=T and B=O to conform with Rabiner's tutorial.
"""

"""
class HMM:
    def __init__(self):
        pass

    def simulate(self,nSteps):

        def drawFrom(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]

        observations = np.zeros(nSteps)
        states = np.zeros(nSteps)
        states[0] = drawFrom(self.pi)
        observations[0] = drawFrom(self.B[states[0],:])
        for t in range(1,nSteps):
            states[t] = drawFrom(self.A[states[t-1],:])
            observations[t] = drawFrom(self.B[states[t],:])
        return observations,states


    def train(self,observations,criterion,graphics=False):
        if graphics:
            plt.ion()

        nStates = self.A.shape[0]
        nSamples = len(observations)

        A = self.A
        B = self.B
        pi = copy(self.pi)
        
        done = False
        while not done:

            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = np.zeros((nStates,nSamples))
            c = np.zeros(nSamples) #scale factors
            alpha[:,0] = pi.T * self.B[:,observations[0]]
            c[0] = 1.0/np.sum(alpha[:,0])
            alpha[:,0] = c[0] * alpha[:,0]
            # Update alpha for each observation step
            for t in range(1,nSamples):
                alpha[:,t] = np.dot(alpha[:,t-1].T, self.A).T * self.B[:,observations[t]]
                c[t] = 1.0/np.sum(alpha[:,t])
                alpha[:,t] = c[t] * alpha[:,t]

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = np.zeros((nStates,nSamples))
            beta[:,nSamples-1] = 1
            beta[:,nSamples-1] = c[nSamples-1] * beta[:,nSamples-1]
            # Update beta backwards from end of sequence
            for t in range(len(observations)-1,0,-1):
                beta[:,t-1] = np.dot(self.A, (self.B[:,observations[t]] * beta[:,t]))
                beta[:,t-1] = c[t-1] * beta[:,t-1]

            xi = np.zeros((nStates,nStates,nSamples-1));
            for t in range(nSamples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T,
                               beta[:,t+1])
                for i in range(nStates):
                    numer = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * \
                            beta[:,t+1].T
                    xi[i,:,t] = numer / denom
  
            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi,axis=1))
            # Need final gamma element for new B
            prod =  (alpha[:,nSamples-1] * beta[:,nSamples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!

            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            newB = copy(B)

            if graphics:
                plt.subplot(2,1,1)
                plt.cla()
                #plt.plot(gamma.T)
                plt.plot(gamma[1])
                plt.ylim(-0.1,1.1)
                plt.legend(('Probability State=1'))
                plt.xlabel('Time')
                plt.draw()
            
            numLevels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            for lev in range(numLevels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma

            if np.max(abs(pi - newpi)) < criterion and \
                   np.max(abs(A - newA)) < criterion and \
                   np.max(abs(B - newB)) < criterion:
                done = 1;
  
            A[:],B[:],pi[:] = newA,newB,newpi

        self.A[:] = newA
        self.B[:] = newB
        self.pi[:] = newpi
        self.gamma = gamma
        

if __name__ == '__main__':
    np.set_printoptions(precision=3,suppress=True)
    if True:
        #'Two states, three possible observations in a state'

        hmm = HMM()
        hmm.pi = np.array([0.5, 0.5])
        hmm.A = np.array([[0.85, 0.15],
                          [0.12, 0.88]])
        hmm.B = np.array([[0.8, 0.1, 0.1],
                          [0.0, 0.0, 1]])

        hmmguess = HMM()
        hmmguess.pi = np.array([0.5, 0.5])
        hmmguess.A = np.array([[0.5, 0.5],
                               [0.5, 0.5]])
        hmmguess.B = np.array([[0.3, 0.3, 0.4],
                               [0.2, 0.5, 0.3]])
    else:
        #three states
        print "Error....this example with three states is not working correctly."
        hmm = HMM()
        hmm.pi = np.array([0.1, 0.4, 0.5])
        hmm.A = np.array([[0.7, 0.2, 0.1],
                          [0.1, 0.6, 0.3],
                          [0.4, 0.2, 0.4]])
        hmm.B = np.array([[0.5, 0.3, 0.2],
                          [0.1, 0.6, 0.3],
                          [0.0, 0.3, 0.7]])
        hmmguess = HMM()
        hmmguess.pi = np.array([0.333, 0.333, 0.333])
        hmmguess.A = np.array([[0.3333, 0.3333, 0.3333],
                               [0.3333, 0.3333, 0.3333],
                               [0.3333, 0.3333, 0.3333]])
        hmmguess.B = np.array([[0.3, 0.3, 0.4],
                               [0.2, 0.5, 0.3],
                               [0.3, 0.3, 0.4]])

    o,s = hmm.simulate(1000)
    hmmguess.train(o,0.0001,graphics=True)

    print 'Actual probabilities\n',hmm.pi
    print 'Estimated initial probabilities\n',hmmguess.pi

    print 'Actual state transition probabililities\n',hmm.A
    print 'Estimated state transition probabililities\n',hmmguess.A

    print 'Actual observation probabililities\n',hmm.B
    print 'Estimated observation probabililities\n',hmmguess.B

    plt.subplot(2,1,2)
    plt.cla()
    plt.plot(np.vstack((s*0.9+0.05,hmmguess.gamma[1,:])).T,'-o',alpha=0.7)
    plt.legend(('True State','Guessed Probability of State=1'))
    plt.ylim(-0.1,1.1)
    plt.xlabel('Time')
    plt.draw()

"""





