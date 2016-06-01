import numpy as np

def runkalman(y, RQratio=10., meanwindow=10):
	"""
	A simple vectorised 1D Kalman smoother

	y
		Input array. Smoothing is always applied over the first 
		dimension

	RQratio	
		An estimate of the ratio of the variance of the output 
		to the variance of the state. A higher RQ ratio will 
		result in more smoothing.

	meanwindow
		The initial mean and variance of the output are 
		estimated over this number of timepoints from the start 
		of the array

	References:
		Ghahramani, Z., & Hinton, G. E. (1996). Parameter Estimation for
		Linear Dynamical Systems.
		http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.55.5997

		Yu, B., Shenoy, K., & Sahani, M. (2004). Derivation of Extented
		Kalman Filtering and Smoothing Equations.
		http://www-npl.stanford.edu/~byronyu/papers/derive_eks.pdf
	"""

	x,vpre,vpost = forwardfilter(y, RQratio=RQratio, meanwindow=meanwindow)
	x,v = backpass(x, vpre, vpost)
	x[np.isnan(x)] = 0.
	return x

def forwardfilter(y, RQratio=10., meanwindow=10):
	"""
	the Kalman forward (filter) pass:
	
		xpost,Vpre,Vpost = forwardfilter(y)
	"""
	
	y = np.array(y,copy=False,subok=True,dtype=np.float32)

	xpre = np.empty_like(y)
	xpost = np.empty_like(y)
	Vpre = np.empty_like(y)
	Vpost = np.empty_like(y)
	K = np.empty_like(y)

	# initial conditions
	pi0 = y[:meanwindow].mean(0)
	ystd = np.std(y,0)
	R = ystd * ystd
	Q = R / RQratio
	V0 = Q

	xpre[0] = xpost[0] = pi0
	Vpre[0] = Vpost[0] = V0
	K[0] = 0

	# loop forwards through time
	for tt in xrange(1, y.shape[0]):

		xpre[tt] = xpost[tt-1]		# is this right? 1/9/13
		# xpre[tt] = xpre[tt-1]

		Vpre[tt] = Vpost[tt-1] + Q	# is this right? 1/9/13
		# Vpre[tt] = Vpre[tt-1] + Q

		K[tt] = Vpre[tt] / (Vpre[tt] + R)
		xpost[tt] = xpre[tt] + K[tt] * (y[tt] - xpre[tt])
		Vpost[tt] = Vpre[tt] - K[tt] * (Vpre[tt])

	return xpost,Vpre,Vpost

def backpass(x, Vpre, V):
	"""	
	the Kalman backward (smoothing) pass:
	
		xpost,Vpost = backpass(x,Vpre,V)
	"""

	xpost = np.empty_like(x)
	Vpost = np.empty_like(x)
	J = np.empty_like(x)

	xpost[-1] = x[-1]
	Vpost[-1] = V[-1]

	# loop backwards through time
	for tt in xrange(x.shape[0]-1, 0, -1):
		J[tt-1] = V[tt-1] / Vpre[tt]
		xpost[tt-1] = x[tt-1] + J[tt-1] * (xpost[tt] - x[tt-1])
		Vpost[tt-1] = V[tt-1] + J[tt-1] * (Vpost[tt] - Vpre[tt]) * J[tt-1]

	return xpost,Vpost