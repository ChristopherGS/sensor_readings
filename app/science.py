# app/science.py

import pandas as pd
import os
# from sklearn import hmm
from hmmlearn.hmm import GaussianHMM
import numpy as np

_basedir = os.path.abspath(os.path.dirname(__file__))
UPLOADS = 'api/uploads'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)

filename = 'badboy3.csv'

PLAY_DATA = UPLOAD_FOLDER + '/' + filename

df = pd.read_csv(PLAY_DATA)

print df


def sql_to_pandas():
	pass

def pandas_cleanup(df):
	columns = []
	df_clean = df[['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'timestamp', 'experiment_id']]
	return df_clean


class HMM:
	def __init__(self):
		pass




def my_hmm():
	pass

	"""
	Probability of being in a particular state at step i is known once we know
	what state we were in at step i-1

	Probability of seeing a particular emission
	at step i is known once we know what state we were in at step i.
	"""

	# how many states will we have?

	# => The observed states are the sensor reading sequences

	# => The hidden state is punch / not punch (or eventually any number of motions)

	# => Emissions encode sensor readings (observed)

	# => states encode motion (hidden)



	# in each state, how many observations can we take? -- all the sensors

	# what will the transition matrix look like?

	# what will the observation probabilities look like?

	# ---------------------------------------------------

	"""
	a sequence of observable X variable is generated 
	by a sequence of internal hidden state Z.
	"""
	# Prepare parameters for a 2-components HMM ###QUESTION 1 - would 'components' reflect the number of motions? 
	# i.e. does component = state
	# Initial population probability

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

	"""List of array-like observation sequences, 
	each of which has shape (n_i, n_features), where n_i is the length of the i_th observation.
	"""

	import datetime
	from matplotlib.finance import quotes_historical_yahoo_ochl
	from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

	# Downloading the data
	date1 = datetime.date(1995, 1, 1)  # start date
	date2 = datetime.date(2012, 1, 6)  # end date
	# get quotes from yahoo finance
	quotes = quotes_historical_yahoo_ochl("INTC", date1, date2)
	if len(quotes) == 0:
	    raise SystemExit

	# unpack quotes
	dates = np.array([q[0] for q in quotes], dtype=int)
	close_v = np.array([q[2] for q in quotes])
	volume = np.array([q[5] for q in quotes])[1:]

	# take diff of close value
	# this makes len(diff) = len(close_t) - 1
	# therefore, others quantity also need to be shifted
	diff = close_v[1:] - close_v[:-1]
	dates = dates[1:]
	close_v = close_v[1:]

	print diff[1:10]
	print volume[1:10]

	# pack diff and volume for training
	X = np.column_stack([diff, volume])

	###############################################################################
	# Run Gaussian HMM
	print("fitting to HMM and decoding ...")

	# make an HMM instance and execute fit
	model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000).fit(X)

	# predict the optimal sequence of internal hidden state
	hidden_states = model.predict(X)

	print("done\n")

	print hidden_states[1:25]

	"""
	X = df['accelerometer_x'].values
	print X
	print type(X)

	Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

	s1 = np.random.randn(50,1)
	s2 = np.random.randn(50,1)+5

	print type(s2)

	signal = np.concatenate([s1, s2])

	print type(signal)

	X = np.column_stack([s1, s2])
	#_x = zip(X, Y)
	#x = _x.shape

	model2 = hmm.GaussianHMM(2, "full")
	model2.fit(X)
	#GaussianHMM(algorithm='viterbi')
	#Z2 = model2.predict(x)

	#print Z2
	"""

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





