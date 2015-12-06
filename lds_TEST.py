class lds:
	"""
	attempt at implementing a LDS of the form:

	x_{t+1} = A*x_t + w_t
	y_{t} = C*x_t + v_t

	where x_t is the state at time t, y_t is the observation at time t.

	w_t is the state
	v_t is the observation noise

	"""

	# 1 establish parameters

		# State transition matrix
		# input matrix
		# observation matrix
		# state covariance matrix
		# observation noise covariance matrix
		# initial state vector
		# initial state covariance

	# Run readiness checks

		# check parameters, lists etc.
		# check if LDS is stable (check eigenvalues)


	# create state transition matrix - will depend on input type
	# synthesize observations - depends on available data / noise
	# returns matrix of observations and matrix of 'tau-dimensional' state vectors

	# Q1 - what is the purpose of tau here?
	# Q2 - what is a 'stable' LDS?
	# Q3 - how does one go about creating a state transition matrix?

