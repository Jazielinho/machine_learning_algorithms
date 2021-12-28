import numpy as np
from sklearn.model_selection import train_test_split

from typing import Tuple

'''
GENERANDO DATOS
'''

def generate_data():
  X = np.concatenate(
      [
       np.random.multivariate_normal(
           mean=[0, 0],
           cov=[[1, 0.5], [0.5, 1]],
           size=1000
           ),
       np.random.multivariate_normal(
           mean=[2, 0],
           cov=[[1, 0.7], [0.7, 1]],
           size=1000
           )
      ],
      axis=0
  )
  y = np.array([1] * 1000 + [-1] * 1000)

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42
  )

  return X_train, X_test, y_train , y_test


X_train, X_test, y_train, y_test = generate_data()


''' SVM CLASE '''
class LinearSVM:
    def __init__(self, learning_rate: float = 0.1, lambda_parameter: float = 0.1, num_iter: int = 100):
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_parameter
        self.num_iter = num_iter
        
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError('Model not fitted, yet.')
        return np.dot(X, self.w) + self.b
    

''' GENERANDO CADA FUNCION '''

def get_init_parameters(n_col: int) -> Tuple[np.array, float]:
    ''' random parameters for w and b '''
    w = np.random.normal(size=n_col)
    b = np.random.normal(size=1)
    return w, b


def get_positive_gradient(x_i: np.array, y_i: float, w: np.array, b: float, lambda_parameter: float) -> Tuple[np.array, float]:
    ''' Positive gradient for SVM'''
    dw = 2 * lambda_parameter * w
    db = 0.0
    return dw, db


def get_negative_gradient(x_i: np.array, y_i: float, w: np.array, b: float, lambda_parameter: float) -> Tuple[np.array, float]:
    ''' Negative gradient for SVM'''
    dw = 2 * lambda_parameter * w - x_i * y_i
    db = y_i
    return dw, db


def get_gradient(x_i: np.array, y_i: float, w: np.array, b: float, lambda_parameter: float) -> Tuple[np.array, float]:
    ''' Gradient for SVM '''
    condition = y_i * np.dot(x_i, w) - b >= 1
    if condition:
        return get_positive_gradient(x_i=x_i, y_i=y_i, w=w, b=b, lambda_parameter=lambda_parameter)
    else:
        return get_negative_gradient(x_i=x_i, y_i=y_i, w=w, b=b, lambda_parameter=lambda_parameter)


def update_parameters(X: np.array, y: np.array, w: np.array, b: float, lambda_parameter: float, learning_rate: float) -> Tuple[np.array, float]:
    ''' UPDATE PARAMETERS FOR SVM '''
    n_row = X.shape[0]
    
    for index in range(n_row):
        x_i = X[index, :]
        y_i = y[index]
        
        dw, db = get_gradient(x_i=x_i, y_i=y_i, w=w, b=b, lambda_parameter=lambda_parameter)
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    return w, b


''' AÑADIENDO LAS FUNCIONES A LA CLASE '''
class LinearSVM:
    def __init__(self, learning_rate: float = 0.1, lambda_parameter: float = 0.1, num_iter: int = 100):
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_parameter
        self.num_iter = num_iter
        
        self.w = None
        self.b = None
    
    def fit(self, X: np.array, y: np.array) -> 'LinearSVM':
        n_col = X.shape[1]
        
        w, b = get_init_parameters(n_col=n_col)
        
        for _ in range(self.num_iter):
            w, b = update_parameters(X=X, y=y, w=w, b=b, lambda_parameter=self.lambda_parameter, learning_rate=self.learning_rate)
        
        self.w = w
        self.b = b
        
        return self
    
    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError('Model not fitted, yet.')
        return np.dot(X, self.w) + self.b

    