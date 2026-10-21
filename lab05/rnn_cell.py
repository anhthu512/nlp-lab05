"""NumPy RNN cell. The student's forward-pass explanation belongs in the scan."""
import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(hidden_size)
        self.W_xh = rng.uniform(-bound, bound, (hidden_size, input_size))
        self.W_hh = rng.uniform(-bound, bound, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size, dtype=np.float64)

    def forward(self, x_t, h_prev):
        x_t = np.asarray(x_t, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)
        if x_t.shape[-1] != self.input_size or h_prev.shape[-1] != self.hidden_size:
            raise ValueError("Input or hidden-state dimension mismatch")
        if x_t.shape[:-1] != h_prev.shape[:-1]:
            raise ValueError("Input and hidden-state batch dimensions must match")
        return np.tanh(x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h)
