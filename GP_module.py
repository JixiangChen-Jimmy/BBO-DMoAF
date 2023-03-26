import GPy
from GPyOpt.util.general import get_quantiles
from scipy.stats import norm
import numpy as np
from math import pow, log, sqrt

class GP:
    def __init__(self, train_x, train_y, lb, ub, k, num_init, nK = 10):

        self.mean = np.mean(train_y)
        self.std = np.std(train_y)

        self.train_x = train_x.copy()
        self.train_y = (train_y - self.mean) / self.std
        self.lb = np.reshape(lb, (1,-1))
        self.ub = np.reshape(ub, (1,-1))
        self.num_train = self.train_x.shape[0]
        self.dim = self.train_x.shape[1]
        self.k = k
        self.nK = nK
        self.num_init = num_init
        self.maxes = np.zeros((1,nK))

        kern = GPy.kern.Matern52(input_dim= self.dim, variance= 1., ARD= False)
        noise_var = 0.01*self.std**2
        self.m = GPy.models.GPRegression(self.train_x, self.train_y, kernel= kern, noise_var= noise_var)
        self.m.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning= False)
        self.m.optimize_restarts(num_restarts= 10, verbose= False)

        self.upsilon = 0.5
        self.delta = 0.05
        self.tau = np.min(train_y)

        self.n_samples = 10
        self.sigma0 = self.m.Gaussian_noise[0]
        self.calc_maxes() # Sample y* for MES acquisition function calculation
        self.set_kappa()
        self.negative_variance_num = 0

    def set_kappa(self):
        num_train = self.num_train
        t = 1 + int((num_train - self.num_init + self.k) / self.k)
        self.kappa = sqrt(self.upsilon * 2 * log(pow(t, 2.0 + self.dim / 2.0) * 3 * pow(np.pi, 2) / (3 * self.delta)))

    def update_model(self, train_x, train_y):
        self.mean = np.mean(train_y)
        self.std = np.std(train_y)
        self.train_x = train_x.copy()
        self.train_y = (train_y - self.mean) / self.std
        self.num_train = self.train_x.shape[0]
        self.set_kappa()
        self.m.set_XY(self.train_x, self.train_y)
        self.m.optimize_restarts(num_restarts=10, verbose= False)

    def calc_maxes(self):
        # Sampling y* via Gumbel Sampling, based on the code implemented in matlab and the theory in original paper
        # @inproceedings{wang2017maxvalue,
        #  title = {Max - value
        # Entropy
        # Search
        # for Efficient Bayesian Optimization},
        # author={Wang, Zi and Jegelka, Stefanie},
        # booktitle={International Conference on Machine Learning (ICML)},
        # year={2017}
        # }
        def probf(x):
            return np.prod(norm.cdf((x - meanVector) / varVector))

        def find_between(val, func, funcvals, mgrid, thres):
            min_idx = np.argmin(np.abs(funcvals - val))
            tt = 0
            if (abs(funcvals[min_idx] - val) < thres):
                return mgrid[0][min_idx]
            if funcvals[min_idx] > val:
                left = mgrid[0][min_idx - 1]
                right = mgrid[0][min_idx]
            else:
                while funcvals[min_idx + 1] + thres < val:
                    # Due to the feature of np.argmin(), it returns the smallest index when multiple identical minimum exist in array.
                    # This feature could lead to wrong choice of right, which is the same as left and run into endless loop.
                    min_idx = min_idx + 1
                left = mgrid[0][min_idx]
                right = mgrid[0][min_idx + 1]
            mid = (left + right) / 2
            midval = func(mid)
            while (abs(midval - val) > thres):
                tt = tt + 1
                if midval > val:
                    right = mid
                else:
                    left = mid
                mid = (left + right) / 2
                midval = func(mid)
            return mid

        gridSize = 10000
        Xgrid = np.tile(self.lb, (gridSize, 1)) + np.tile(self.ub - self.lb, (gridSize, 1)) * np.random.rand(gridSize, 1)
        Xgrid = np.concatenate((Xgrid, self.train_x), 0)
        a = np.array([self.predict(x) for x in Xgrid])
        # Transform the minimization of y into maximization of -y
        meanVector = -a[:, 0].reshape(-1, 1)
        varVector = a[:, 1].reshape(-1, 1)
        sx = np.shape(Xgrid)[0]

        left = np.max(-self.train_y)
        if probf(left) < 0.25:
            right = max(meanVector + 5 * varVector)
            while (probf(right) < 0.75):
                right = right + right - left
            mgrid = np.linspace(left, right, 100).reshape(1, -1)
            prob = np.prod(
                norm.cdf((np.tile(mgrid, (sx, 1)) - np.tile(meanVector, (1, 100))) / np.tile(varVector, (1, 100))), 0)
            med = find_between(0.5, probf, prob, mgrid, 0.01)
            q1 = find_between(0.25, probf, prob, mgrid, 0.01)
            q2 = find_between(0.75, probf, prob, mgrid, 0.01)
            beta = (q1 - q2) / (np.log(np.log(4 / 3)) - np.log(np.log(4)))
            alpha = med + beta * np.log(np.log(2))
            self.maxes = -np.log(-np.log(np.random.rand(1, self.nK))) * beta + alpha
            self.maxes[0][np.argwhere(self.maxes < left + 5 * np.sqrt(self.sigma0))] = left + 5 * np.sqrt(self.sigma0)
        else:
            self.maxes[0, :] = left + 5 * np.sqrt(self.sigma0)

    def predict(self, x):
        m, v = self.m.predict(x.reshape(1, x.size))
        pys = m[0][0]
        pss = v[0][0]
        pys = self.mean + (pys * self.std)
        pss = np.clip(pss * (self.std ** 2), 1e-20, None)
        return pys, np.sqrt(pss)

    def MES(self, x, pys, pss):
        meanVector = np.array([-pys]).reshape(-1, 1)
        meanVector = np.tile(meanVector, (1, self.nK))
        varVector = np.array([pss]).reshape(-1, 1)
        varVector = np.tile(varVector, (1, self.nK))
        gamma = (self.maxes - meanVector) / varVector
        pdfgamma = norm.pdf(gamma)
        cdfgamma = np.clip(norm.cdf(gamma), 1e-20, None)
        return np.mean(gamma * pdfgamma / (2 * cdfgamma) - np.log(cdfgamma))

    def GP_LCB(self, x, pys, pss):
        lcb = pys - self.kappa * pss
        return lcb

    def EI(self, x, pys, pss, eps):
        phi, Phi, u = get_quantiles(eps, self.tau, pys, pss)
        ei = pss * (u * Phi + phi)
        return ei

    def PI(self, x, pys, pss, eps):
        _, Phi, _ = get_quantiles(eps, self.tau, pys, pss)
        pi = Phi
        return pi

    def MACE_8(self, x):
        pys, pss = self.predict(x)
        ei_0 = self.EI(x, pys, pss, 1e-3)
        ei_1 = self.EI(x, pys, pss, 1e-1)
        ei_2 = self.EI(x, pys, pss, 10)
        pi_0 = self.PI(x, pys, pss, 1e-3)
        pi_1 = self.PI(x, pys, pss, 1e-1)
        pi_2 = self.PI(x, pys, pss, 10)
        gp_lcb = self.GP_LCB(x, pys, pss)
        mes = self.MES(x, pys, pss)
        return ei_0, ei_1, ei_2, pi_0, pi_1, pi_2, gp_lcb, mes