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

	def sigmoid(self, a):
		return (1 / (1 + np.exp(-a)))
	
	def tanh(self, a):
		return np.tanh(a)

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		seq_len, feature_len = x.shape 
		if feature_len != self.input_size:
			raise ValueError("Input shape in dimension -1 should match LSTM Input Size")
		hidden_state = np.asarray(initial_hidden_state, dtype=float).reshape(self.hidden_size, 1)
		cell_state = np.asarray(initial_cell_state, dtype=float).reshape(self.hidden_size, 1)
		outputs = []
		for x_i in x: 
			x_h_t = np.vstack((hidden_state, x_i.reshape(-1, 1))) # (hidden_size + input_size, 1)
			forget_gate_output = self.sigmoid(self.Wf @ x_h_t + self.bf)
			input_gate_output = self.sigmoid(self.Wi @ x_h_t + self.bi)
			cell_gate_output = self.tanh(self.Wc @ x_h_t + self.bc)
			output_gate_output = self.sigmoid(self.Wo @ x_h_t + self.bo)
			cell_state = forget_gate_output * cell_state + input_gate_output * cell_gate_output
			hidden_state = output_gate_output * self.tanh(cell_state)
			outputs.append(hidden_state)
		return np.array(outputs), hidden_state, cell_state 
