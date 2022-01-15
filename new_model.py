
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import optuna

from typing import List


class LinearCooperativeLearningTf(tf.keras.models.Model):
    def __init__(self, lambda_regularization: float):
        super().__init__(self)
        self.linear_x = tf.keras.layers.Dense(
            units=1,
            kernel_regularizer=tf.keras.regularizers.l1(lambda_regularization),
            use_bias=False,
            name='X'
        )
        self.linear_z = tf.keras.layers.Dense(
            units=1,
            kernel_regularizer=tf.keras.regularizers.l1(lambda_regularization),
            use_bias=False,
            name = 'Z'
        )

    def call(self, inputs: List) -> List:
        return [self.linear_x(inputs[0]), self.linear_z(inputs[1])]


def mean_squared_error_cooperative_learning_loss(rho_regularization: float):
    def _contrastive_loss(linear_x, linear_z):
        return tf.square(linear_x - linear_z)

    def _squared_error(y_true, linear_x, linear_z):
        return tf.square(y_true - linear_x - linear_z)

    def loss(y_true, y_predict):
        y_true = tf.reshape(y_true, (-1, 1))
        y_true = tf.cast(y_true, tf.float32)
        linear_x, linear_z = y_predict
        linear_x = tf.reshape(linear_x, (-1, 1))
        linear_z = tf.reshape(linear_z, (-1, 1))
        linear_x = tf.cast(linear_x, tf.float32)
        linear_z = tf.cast(linear_z, tf.float32)
        squared_error = _squared_error(y_true, linear_x, linear_z)
        contrastive_loss = _contrastive_loss(linear_x, linear_z)
        _loss = (1/2) * squared_error + (rho_regularization / 2) * contrastive_loss
        return tf.reduce_mean(_loss)

    return loss


class LinearCooperativeLearning:
    def __init__(self, rho_regularization, lambda_regularization, num_iter):
        self.rho_regularization = rho_regularization
        self.lambda_regularization = lambda_regularization
        self.num_iter = num_iter
        self.tf_model = LinearCooperativeLearningTf(lambda_regularization=self.lambda_regularization)
        self.loss = mean_squared_error_cooperative_learning_loss(rho_regularization=self.rho_regularization)
        self.optimizer = tf.keras.optimizers.Adam(lr=0.1)
        self.list_loss = []

    def fit(self, X, Z, y):
        list_loss = []
        for epoch in tqdm.tqdm(range(self.num_iter)):
            with tf.GradientTape() as tape:
                y_predict = self.tf_model([X, Z], training=True)
                loss_value = self.loss(y, y_predict)
            grads = tape.gradient(loss_value, self.tf_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.tf_model.trainable_variables))
            list_loss.append(loss_value.numpy())
        self.list_loss = list_loss

    def predict(self, X, Z):
        linear_x, linear_z = self.tf_model.predict([X, Z])
        return (linear_x + linear_z).flatten()

    def coef(self):
        weights_X = self.tf_model.get_layer('X').get_weights()[0]
        weights_Z = self.tf_model.get_layer('Z').get_weights()[0]
        return {'weights_X': weights_X, 'weights_Z': weights_Z}


def create_dataset(n=200, p=1000, snr=1.2, s_u=1, t=6, b_x=2, b_z=0, sigma=1):
    x_i = [np.random.normal(0, 1, size=(n, 1)) for _ in range(p)]
    z_i = [np.random.normal(0, 1, size=(n, 1)) for _ in range(p)]

    p_r = int(snr * p / (1 + snr))

    for i in range(p_r):
        u_i = np.random.normal(0, s_u, size=(n, 1))
        x_i[i] = x_i[i] + t * u_i
        z_i[i] = z_i[i] + t * u_i

    X = np.concatenate(x_i, axis=1)
    Z = np.concatenate(z_i, axis=1)

    beta_x = [b_x for _ in range(p_r)] + [0 for _ in range(p - p_r)]
    beta_z = [b_z for _ in range(p_r)] + [0 for _ in range(p - p_r)]

    y = X @ beta_x + Z @ beta_z + np.random.normal(0, sigma, size=n)

    return X, Z, y


class LinearCooperativeRegularizedRegression:
    def __init__(self, lambda_regularization=5.0, rho_regularization=0.5):
        self.lambda_regularization = lambda_regularization
        self.rho_regularization = rho_regularization
        self.scale_X = StandardScaler(with_mean=False, with_std=True)
        self.scale_Z = StandardScaler(with_mean=False, with_std=True)
        self.scale_y = StandardScaler(with_mean=True, with_std=False)
        self.en = ElasticNet(fit_intercept=False, l1_ratio=1, alpha=lambda_regularization, max_iter=1e5)

    def _prepare_X_Z(self, X, Z):
        X_1 = np.concatenate([X, Z], axis=1)
        X_2 = np.concatenate([(-1) * self.rho_regularization ** 0.5 * X,
                              self.rho_regularization ** 0.5 * Z], axis=1)
        X_ = np.concatenate([X_1, X_2], axis=0)
        return X_

    def fit(self, X, Z, y):
        y_reshape = y.reshape(-1, 1)
        scale_X = self.scale_X.fit_transform(X)
        scale_Z = self.scale_Z.fit_transform(Z)
        center_y = self.scale_y.fit_transform(y_reshape).flatten()

        X_ = self._prepare_X_Z(scale_X, scale_Z)
        y_ = np.concatenate([center_y, [0 for _ in range(X.shape[0])]])
        self.en.fit(X_, y_)
        return self

    def predict(self, X, Z):
        scale_X = self.scale_X.transform(X)
        scale_Z = self.scale_Z.transform(Z)

        X_ = self._prepare_X_Z(scale_X, scale_Z)
        y_predict_ = self.en.predict(X_)
        y_predict = y_predict_[:X.shape[0]]
        return self.scale_y.inverse_transform(y_predict.reshape(-1, 1)).flatten()


def optim(X, Z, y, X_test, Z_test, y_test, rho_regularization=None, timeout=60, n_trials=1000, show_progress_bar=True):

    def objective(trial):
        if rho_regularization is None:
            rho_regularization_ = trial.suggest_uniform('rho_regularization', 0, 10)
        else:
            rho_regularization_ = rho_regularization

        lambda_regularization = trial.suggest_uniform('lambda_regularization', 0, 100)
        lr = LinearCooperativeRegularizedRegression(lambda_regularization=lambda_regularization,
                                                    rho_regularization=rho_regularization_)
        lr.fit(X, Z, y)
        predictions = lr.predict(X_test, Z_test)
        error = mean_squared_error(y_test, predictions)
        return error

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_jobs=-1, timeout=timeout, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best_params = study.best_params

    return best_params



if __name__ == '__main__':
    # X, Z, y = create_dataset(n=500, p=200, snr=1.1, s_u=1, t=2, b_x=1.5, b_z=3, sigma=1)#
    # X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.1, s_u=1, t=2, b_x=1.5, b_z=3, sigma=1)
    # X, Z, y = create_dataset(n=500, p=200, snr=0.3, s_u=1, t=0, b_x=2, b_z=2, sigma=1)
    # X_test, Z_test, y_test = create_dataset(n=100, p=200, snr=0.3, s_u=1, t=0, b_x=2, b_z=2, sigma=1)
    X, Z, y = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
    X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)

    param_0 = optim(X, Z, y, X_test, Z_test, y_test, rho_regularization=0, show_progress_bar=False)
    param_1 = optim(X, Z, y, X_test, Z_test, y_test, show_progress_bar=False)
    param_2 = optim(X, Z, y, X_test, Z_test, y_test, rho_regularization=1, show_progress_bar=False)

    list_error_0 = []
    list_error_1 = []
    list_error_2 = []

    self_0 = LinearCooperativeRegularizedRegression(lambda_regularization=param_0['lambda_regularization'], rho_regularization=0)
    self_0.fit(X, Z, y)

    self_1 = LinearCooperativeRegularizedRegression(lambda_regularization=param_1['lambda_regularization'], rho_regularization=param_1['rho_regularization'])
    self_1.fit(X, Z, y)

    self_2 = LinearCooperativeRegularizedRegression(lambda_regularization=param_2['lambda_regularization'], rho_regularization=1)
    self_2.fit(X, Z, y)

    for _ in tqdm.tqdm(range(100)):
        X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
        error_0 = mean_squared_error(y_test, self_0.predict(X_test, Z_test))
        error_1 = mean_squared_error(y_test, self_1.predict(X_test, Z_test))
        error_2 = mean_squared_error(y_test, self_2.predict(X_test, Z_test))

        list_error_0.append(error_0)
        list_error_1.append(error_1)
        list_error_2.append(error_2)

    df = pd.DataFrame({'error_0': list_error_0, 'error_1': list_error_1, 'error_2': list_error_2})
    df.plot.box()

