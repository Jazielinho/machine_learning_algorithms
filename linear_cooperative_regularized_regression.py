

import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import optuna
from typing import Tuple, Dict, Optional


def create_dataset(n: int = 200, p: int = 1000, snr: float = 1.2, s_u: float = 1, t: float = 6,
                   b_x: float = 2, b_z: float = 0, sigma: float = 1) -> Tuple[np.array, np.array, np.array]:
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
    def __init__(self, lambda_regularization: float = 5.0, rho_regularization: float = 0.5):
        self.lambda_regularization = lambda_regularization
        self.rho_regularization = rho_regularization
        self.scale_X = StandardScaler(with_mean=True, with_std=False)
        self.scale_Z = StandardScaler(with_mean=True, with_std=False)
        self.scale_y = StandardScaler(with_mean=True, with_std=False)
        self.en = ElasticNet(fit_intercept=False, l1_ratio=1, alpha=lambda_regularization, max_iter=1e5)

    def _prepare_X_Z(self, X: np.array, Z: np.array) -> np.array:
        X_1 = np.concatenate([X, Z], axis=1)
        X_2 = np.concatenate([(-1) * self.rho_regularization ** 0.5 * X,
                              self.rho_regularization ** 0.5 * Z], axis=1)
        X_ = np.concatenate([X_1, X_2], axis=0)
        return X_

    def fit(self, X: np.array, Z: np.array, y: np.array):
        y_reshape = y.reshape(-1, 1)
        scale_X = self.scale_X.fit_transform(X)
        scale_Z = self.scale_Z.fit_transform(Z)
        center_y = self.scale_y.fit_transform(y_reshape).flatten()

        X_ = self._prepare_X_Z(scale_X, scale_Z)
        y_ = np.concatenate([center_y, [0 for _ in range(X.shape[0])]])
        self.en.fit(X_, y_)
        return self

    def predict(self, X: np.array, Z: np.array) -> np.array:
        scale_X = self.scale_X.transform(X)
        scale_Z = self.scale_Z.transform(Z)

        X_ = self._prepare_X_Z(scale_X, scale_Z)
        y_predict_ = self.en.predict(X_)
        y_predict = y_predict_[:X.shape[0]]
        return self.scale_y.inverse_transform(y_predict.reshape(-1, 1)).flatten()


class _LinearCooperativeRegularizedRegression:
    def __init__(self, lambda_regularization: float = 5.0, rho_regularization: float = 0.5):
        self.lambda_regularization = lambda_regularization
        self.rho_regularization = rho_regularization
        self.scale_X = StandardScaler(with_mean=True, with_std=False)
        self.scale_Z = StandardScaler(with_mean=True, with_std=False)
        self.scale_y = StandardScaler(with_mean=True, with_std=False)
        self.en_X = ElasticNet(fit_intercept=False, l1_ratio=1, alpha=lambda_regularization, max_iter=1e5)
        self.en_Z = ElasticNet(fit_intercept=False, l1_ratio=1, alpha=lambda_regularization, max_iter=1e5)

    def fit(self, X: np.array, Z: np.array, y: np.array):
        y_reshape = y.reshape(-1, 1)
        scale_X = self.scale_X.fit_transform(X)
        scale_Z = self.scale_Z.fit_transform(Z)
        center_y = self.scale_y.fit_transform(y_reshape).flatten()

        coef_Z = np.zeros(shape=scale_Z.shape[1])

        coef_X_old = np.zeros(shape=scale_X.shape[1])
        coef_Z_old = np.zeros(shape=scale_Z.shape[1])

        iter = 0

        while True:
            y_x = center_y / (1 + self.rho_regularization) - (1 - self.rho_regularization) * scale_Z @ coef_Z / (1 + self.rho_regularization)
            self.en_X.fit(scale_X, y_x)
            coef_X = self.en_X.coef_

            y_z = center_y / (1 + self.rho_regularization) - (1 - self.rho_regularization) * scale_X @ coef_X / (1 + self.rho_regularization)
            self.en_Z.fit(scale_Z, y_z)
            coef_Z = self.en_Z.coef_

            dif = ((np.sum((coef_X - coef_X_old) ** 2)) ** 0.5 + (np.sum((coef_Z - coef_Z_old) ** 2)) ** 0.5) / 2

            if dif < 1e-4:
                break
            else:
                coef_X_old = coef_X.copy()
                coef_Z_old = coef_Z.copy()
                iter += 1
            if iter > 99:
                break

        return self

    def predict(self, X: np.array, Z: np.array) -> np.array:
        scale_X = self.scale_X.transform(X)
        scale_Z = self.scale_Z.transform(Z)

        y_predict_ = self.en_X.predict(scale_X) + self.en_Z.predict(scale_Z)
        return self.scale_y.inverse_transform(y_predict_.reshape(-1, 1)).flatten()


def optim_linear_cooperative(X: np.array, Z: np.array, y: np.array,
                             X_test: np.array, Z_test: np.array, y_test: np.array,
                             rho_regularization: Optional[float] = None, timeout: int = 60, n_trials: int = 1000,
                             show_progress_bar: bool = True) -> Dict:

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


def optim_lasso(X: np.array, Z: np.array, y: np.array,
                X_test: np.array, Z_test: np.array, y_test: np.array,
                timeout: int = 60, n_trials: int = 1000, show_progress_bar: bool = True):
    X_train_ = np.concatenate([X, Z], axis=1)
    X_test_ = np.concatenate([X_test, Z_test], axis=1)

    def objective(trial):
        alpha = trial.suggest_uniform('alpha', 0, 100)
        lr = ElasticNet(alpha=alpha, max_iter=1e5, l1_ratio=1)
        lr.fit(X_train_, y)
        predictions = lr.predict(X_test_)
        error = mean_squared_error(y_test, predictions)
        return error

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_jobs=-1, timeout=timeout, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best_params = study.best_params

    return best_params


if __name__ == '__main__':
    # X, Z, y = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
    # X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
    X, Z, y = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
    X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
    # X, Z, y = create_dataset(n=500, p=200, snr=0.3, s_u=1, t=0, b_x=2, b_z=2, sigma=1)
    # X_test, Z_test, y_test = create_dataset(n=100, p=200, snr=0.3, s_u=1, t=0, b_x=2, b_z=2, sigma=1)
    # X, Z, y = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
    # X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)

    param_0 = optim_linear_cooperative(X, Z, y, X_test, Z_test, y_test, rho_regularization=0, show_progress_bar=False)
    param_1 = optim_linear_cooperative(X, Z, y, X_test, Z_test, y_test, show_progress_bar=False)
    param_2 = optim_linear_cooperative(X, Z, y, X_test, Z_test, y_test, rho_regularization=1, show_progress_bar=False)
    param_lasso = optim_lasso(X, Z, y, X_test, Z_test, y_test, show_progress_bar=False)

    list_error_0 = []
    list_error_1 = []
    list_error_2 = []
    list_error_lasso = []

    self_0 = LinearCooperativeRegularizedRegression(lambda_regularization=param_0['lambda_regularization'], rho_regularization=0)
    self_0.fit(X, Z, y)

    self_1 = LinearCooperativeRegularizedRegression(lambda_regularization=param_1['lambda_regularization'], rho_regularization=param_1['rho_regularization'])
    self_1.fit(X, Z, y)

    self_2 = LinearCooperativeRegularizedRegression(lambda_regularization=param_2['lambda_regularization'], rho_regularization=1)
    self_2.fit(X, Z, y)

    self_lasso = ElasticNet(alpha=param_lasso['alpha'], max_iter=1e5, l1_ratio=1)
    self_lasso.fit(np.concatenate([X, Z], axis=1), y)

    for _ in tqdm.tqdm(range(100)):
        X_test, Z_test, y_test = create_dataset(n=500, p=200, snr=1.0, s_u=1, t=6, b_x=2, b_z=2, sigma=1)
        error_0 = mean_squared_error(y_test, self_0.predict(X_test, Z_test))
        error_1 = mean_squared_error(y_test, self_1.predict(X_test, Z_test))
        error_2 = mean_squared_error(y_test, self_2.predict(X_test, Z_test))
        error_lasso = mean_squared_error(y_test, self_lasso.predict(np.concatenate([X_test, Z_test], axis=1)))

        list_error_0.append(error_0)
        list_error_1.append(error_1)
        list_error_2.append(error_2)
        list_error_lasso.append(error_lasso)

    df = pd.DataFrame({'error_0': list_error_0, 'error_1': list_error_1, 'error_2': list_error_2, 'error_lasso': list_error_lasso})
    df.plot.box()

