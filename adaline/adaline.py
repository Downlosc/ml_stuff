import numpy as np 

class Adaline: 

  def __init__(self, eta = 0.01, epochs = 100, random_state = 1): 
    self.eta          = eta
    self.epochs       = epochs
    self.random_state = random_state
    self.weights      = []
    self.cost_vector  = []

  def net_input(self, X): 
    weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
    return weighted_sum

  def initialize_weights(self, X):
    generator    = np.random.RandomState(self.random_state) 
    self.weights = generator.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])

  def predict(self, X): 
    return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)

  # Linear activation | just an identity function
  def activation_function(self, inpt):
    return inpt

  def fit(self, X, y): 
    self.initialize_weights(X)
    for epoch in range(self.epochs):
      # Batch gradient descending 
      inpt   = self.net_input(X)
      output = self.activation_function(inpt)

      errors = y - output

      gradient      = self.eta * (X.T.dot(errors))
      gradient_bias = self.eta * errors.sum()

      self.weights[1:] += gradient
      self.weights[0]  += gradient_bias

      
      # Keep track of costs 
      cost = (errors ** 2).sum() / 2.0
      self.cost_vector.append(cost)
  
  def score(self, X, y): 
    missed = 0 
    for xi, yi in zip(X, y): 
      y_hat = self.predict(xi)
      if (y_hat != yi): missed += 1 
    total = len(X) 
    self.score =  (total - missed / total) 
    return self.score


