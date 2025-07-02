import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):

		h_t = np.array(initial_hidden_state).reshape(-1, 1)
		c_t = np.array(initial_cell_state).reshape(-1, 1)
		
		for x_t in x:
			x_t = np.array(x_t).reshape(-1, 1)
			combined = np.vstack((h_t, x_t))
			f_t = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
			i_t = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
			o_t = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
			c_hat_t = np.tanh(np.dot(self.Wc, combined) + self.bc)
			c_t = f_t * c_t + i_t * c_hat_t
			h_t = o_t * np.tanh(c_t)
			
		return h_t, c_t

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
