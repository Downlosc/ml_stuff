import numpy as np 

def _print_initialize(eta, epochs, random_state, weights): 
  q = """ 
        Initializing class Adaline.. 
        learning rate = %s
        epochs        = %s 
        random_state  = %s 
        random_weights = %s 

    """ % (eta, epochs, random_state, np.array2string(weights, separator=','))

  print(q)


def _print_prediction(xi, yi, y_hat):
  print('Feature vector-->', xi)
  print('Target value -->', yi)
  print('Predicted value:', y_hat)

  print('\n\n\n\n\n')

def _print_score(total, missed, score): 
  q = 'Total predictions = %d \n' % total
  q += 'Missed predictions = %d \n' % missed
  q += 'Score = %f' % score
  print(q)
  return score


class Adaline: 

  def __init__(self, X, y, eta = 0.00005, epochs = 1500, random_state = 52): 
    self.eta          = eta
    self.epochs       = epochs
    self.random_state = random_state
    self.X            = X
    self.y            = y
    self.weights      = self.initialize_weights()
    self.cost_vector  = []

    _print_initialize(eta, epochs, random_state, self.weights)

  def net_input(self, X): 
    weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
    return weighted_sum

  def initialize_weights(self):
    generator    = np.random.RandomState(self.random_state) 
    weights = generator.normal(loc = 0.0, scale = 0.01, size = 1 + self.X.shape[1])
    return weights

  # Linear activation | just an identity function
  def activation_function(self, inpt):
    return inpt

  def predict(self, X): 
    net = self.net_input(X)
    return np.where(self.activation_function(net) >= 0.0, 1, 0)

  def fit(self): 
    for epoch in range(self.epochs):
      # Batch gradient descending 
      inpt   = self.net_input(self.X)
      output = self.activation_function(inpt)

      errors = self.y - output

      gradient      = self.eta * (self.X.T.dot(errors))
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
      _print_prediction(xi, yi, y_hat)
    total = len(X) 
    score =  (total - missed) / total 
    _print_score(total, missed, score)
    return self


