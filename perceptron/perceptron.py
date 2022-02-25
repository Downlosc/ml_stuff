import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron: 
  def __init__(self, eta = 0.01, epochs = 100, random_state = 1): 
    self.eta          = eta
    self.epochs       = epochs
    self.random_state = random_state
    self.weights      = []
    self.errors       = []
    self.score        = None

  def initialize_weights(self, X): 
    generator    = np.random.RandomState(self.random_state)
    self.weights = generator.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])

  def fit(self, X, y): 
    self.initialize_weights(X)
    for niter in range(self.epochs):
      err = 0 
      for xi, yi in zip(X, y): 
        y_hat = self.predict(xi) 
        update = self.eta * (yi - y_hat) 
        self.weights[1:] += update * xi
        self.weights[0]  += update
        err += int(update != 0.0) 
    self.errors.append( err ) 

  def net_input(self, X): 
    weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
    return weighted_sum

  def activation_function(self, X): 
    ws = self.net_input(X)
    return np.where(ws >= 0.0, 1, 0) 

  def predict(self, X): 
    return self.activation_function(X) 

  def score_compute(self, X, y): 
    wrong_predictions = 0
    for xi, yi in zip(X,y):
      prediction = self.predict(xi)
      q = "Prediction = %d  -- target = %d " % (prediction, yi)
      if (yi != prediction): wrong_predictions += 1
    total_predictions = len(X)
    self.score = (total_predictions - wrong_predictions) / total_predictions
    return self.score

    

    
