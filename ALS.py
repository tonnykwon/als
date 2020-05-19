"""
cost_function: return cost
fit: train the model on data
predict: predict labels for given users
"""


import numpy as np
from scipy.sparse import csr_matrix, diags,eye
import sys

class ALS:
	
	def __init__(self, reg = 1e-3, alpha = 40):
	
		self.reg = reg
		self.alpha = alpha
		
	def progress(self, i, n):
		percent = (i+1) /n
		sys.stdout.write('\r')
		sys.stdout.write("[%-50s] %d%%" % ('='*int(50*percent), 100*percent))
		sys.stdout.flush()
		
	def cost_function(self):
		
		main_cost = self.C.multiply((self.R - self.X.dot(self.Y.T)).power(2)).sum()
		reg_term = self.reg*(self.X.power(2).sum() + self.Y.power(2).sum())
		cost = main_cost + reg_term
		return cost
	
		
	def fit(self, R, hidden_size = 100, iteration = 5):
		from scipy.sparse.linalg import inv

		# data(R) must be csr_matrix format n by m
		self.R = R
		left_size, right_size = R.shape
		
		# init
		self.X = csr_matrix(np.random.normal(size = (left_size, hidden_size)))
		self.Y = csr_matrix(np.random.normal(size = (right_size, hidden_size)))
		
		# confidence level
		self.C = csr_matrix(R.copy()*self.alpha + np.array([1]* (left_size*right_size)).reshape((left_size, right_size)))
		
		# cost
		cost = self.cost_function()
		print('cost: '+str(cost))
		
		# iterate
		for iterate in range(iteration):
			print('iter: '+str(iterate))
			print('User Matrix')
			for u, Cu in enumerate(self.C):
				C_diag = diags(Cu.toarray()[0], shape = [right_size, right_size])
				self.X[u] = inv(self.Y.T.dot(C_diag).dot(self.Y) + eye(hidden_size)*self.reg ).dot(self.Y.T).dot(C_diag).dot(R[u].T).T
				
				self.progress(u, left_size)
				
			print('\nMovie Matrix')
			for i, Ci in enumerate(self.C.T):
				C_diag = diags(Ci.toarray()[0], shape = [left_size, left_size])
				self.Y[i] = inv(self.X.T.dot(C_diag).dot(self.X) + eye(hidden_size)*self.reg ).dot(self.X.T).dot(C_diag).dot(R.T[i].T).T
				
				self.progress(i, right_size)

		
			self.cost = self.cost_function()
			print('\ncost: '+str(self.cost))
		
		# save prediciton matrix
		self.prediction = self.X.dot(self.Y.T)
			
		return self.cost
	
	def predict(self, idx, top = 10):
		recommendation = (-self.prediction[idx]).toarray().argsort()[0, :top]
		return recommendation