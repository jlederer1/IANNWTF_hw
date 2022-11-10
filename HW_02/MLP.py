import numpy as np
import tensorflow as tf

class Layer:
  def __init__(self, n_units, input_units):
    self.n_units = n_units
    self.input_units = input_units

    np.random.seed(42)
    self.W = np.random.rand(input_units, n_units)
    self.b = np.zeros(n_units)

    self.n_input = None
    self.n_preactivation = None
    self.n_activation = None

    # learning_rate
    alpha = 0.03

    print("W: ", self.W.shape, "\n", self.W, "\n")
    print("b: ", self.b.shape, "\n", self.b, "\n")

  def relu(self, x): 
    return np.maximum(0, x)

  def forward_step(self, arr):
    self.n_input = arr
    self.n_preactivation = self.n_input * self.W 
    self.n_activation = self.relu(self.n_preactivation)

    print("input: ", self.n_input.shape, "\n", self.n_input, "\n")
    print("preactivation: ", self.n_preactivation.shape, self.n_preactivation, "\n")
    print("activation: ", self.n_activation.shape, self.n_activation, "\n")

    return self.n_activation

  def backward_step(self, targets):
    d_L_d_a_layer = self.n_activation - targets
    d_L_d_d_layer = d_L_d_a_layer * np.where(self.n_activation > 0, 1, 0)
    d_L_d_W_layer = d_L_d_d_layer * self.n_activation
    
    gradient_W_layer = self.n_input.T * (  np.where(self.n_preactivation > 0, 1, 0) @  d_L_d_a_layer)
    gradient_B_layer = np.where(self.n_preactivation > 0, 1, 0) @  d_L_d_a_layer
    gradient_input = ( np.where(self.n_preactivation > 0, 1, 0) * d_L_d_a_layer ) @ W.T

    # update parameters
    self.W = self.W - alpha * ( d_L_d_W_layer )
    self.b = self.b - alpha * ( d_L_d_b_layer )

    print("gradient_W_layer: ", gradient_W_layer, "\n")
    print("gradient_B_layer: ", gradient_B_layer, "\n")
    print("gradient_input: ", gradient_input, "\n")
    print("gradient_input: ", gradient_input, "\n")

    return gradient_input






