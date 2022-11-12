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
    self.alpha = 0.03

    self.loss = np.array([])

  def relu(self, x): 
    return np.maximum(0, x)

  def forward_step(self, arr):
    self.n_input = np.append(arr, 1)
    self.n_preactivation = self.n_input @ np.vstack((self.W, self.b))
    self.n_activation = self.relu(self.n_preactivation)

    return self.n_activation

  def backward_step(self, d_L_d_a_layer):

    if self.n_units == 1:
      loss = ( (self.n_activation -  d_L_d_a_layer) ** 2 ) / 2
      self.loss = np.append(self.loss, loss)
      # print(loss)

    if self.n_units != 1:
      d_L_d_a_layer = self.n_activation -  d_L_d_a_layer

    d_L_d_d_layer = d_L_d_a_layer * np.where(self.n_activation > 0, 1, 0)
    d_L_d_W_layer = d_L_d_d_layer * self.n_activation
    
    gradient_W_layer = self.n_input.T * (  np.where(self.n_preactivation > 0, 1, 0) @  d_L_d_a_layer)
    gradient_b_layer = np.where(self.n_preactivation > 0, 1, 0) @  d_L_d_a_layer
    gradient_input = ( np.where(self.n_preactivation > 0, 1, 0) * d_L_d_a_layer ) @ self.W.T

    # update parameters
    self.W = self.W - self.alpha * ( d_L_d_W_layer )
    self.b = self.b - self.alpha * ( d_L_d_d_layer )

    return gradient_input

class MLP_Gen9000:
  def __init__(self, data):
    self.l_HIDDEN = Layer(10, 1)
    self.l_OUT = Layer(1, 10)

    self.inputs = data[0]
    self.targets = data[1]

  def forward_step(self, x):
    a1 = self.l_HIDDEN.forward_step(x)
    a2 = self.l_OUT.forward_step(np.array([a1]))

    return a2

  def backpropagation(self, target): 
    gradient_input1 = self.l_OUT.backward_step(target)
    gradient_input2 = self.l_HIDDEN.backward_step(gradient_input1)

  def training(self, n_epochs):
    for i in range(n_epochs):
      
      for i in range(len(self.inputs)):
        self.forward_step(self.inputs[i])
        self.backpropagation(np.array([self.targets[i]]))
    
    return self.l_OUT.loss



