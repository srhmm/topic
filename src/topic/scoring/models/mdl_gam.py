import numpy as np
from pygam import GAM

n_splines = 20
order = 3


class GAM_MDL(GAM):

    def __init__(self, n_splines=20, order=3):
        super(GAM, self).__init__(n_splines, order)
        self.n_splines = n_splines
        self.order = order

    def fit(self, X, y):
        super().fit(X, y)
        mse = np.mean((self.predict(X) - y) ** 2)
        n = X.shape[0]
        p = X.shape[1] * self.n_splines * self.order

        self.mdl_lik_train = n * np.log(mse)
        self.mdl_model_train = 2 * p
        self.mdl_pen_train = 0
        self.mdl_train = self.mdl_lik_train + self.mdl_model_train + self.mdl_pen_train

    def mdl_score_ytrain(self):
        return self.mdl_train, self.mdl_lik_train, self.mdl_model_train, self.mdl_pen_train

    def mdl_score_ytest(self, X_test, y_test):
        mse = np.mean((self.predict(X_test) - y_test) ** 2)
        n = X_test.shape[0]
        p = X_test.shape[1] * self.n_splines * self.order

        log_lik = n * np.log(mse)
        m_penalty = 2 * p
        X_penalty = 0
        mdl = log_lik + m_penalty + X_penalty
        return mdl, log_lik, m_penalty, X_penalty


def AIC(mse, n, p):
    return 2 * p + n * np.log(mse)
