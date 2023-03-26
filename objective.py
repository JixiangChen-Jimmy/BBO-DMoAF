import numpy as np
import opfunu

from opfunu.cec_based.cec2015 import F12015,F22015,F32015,F42015,F52015,F62015,F72015,F82015,F92015,F102015,F112015,F122015,F132015,F142015,F152015

# CEC 15的 15个function分别是
# F12015, F22015, F32015, F42015, F52015,
# F62015, F72015, F82015, F92015, F102015,
# F112015, F122015, F132015, F142015, F152015
class CEC_2015_functions():
    def __init__(self, dim=10, func_name = ''):
        self.dim = dim
        self.bounds = [(-100.0, 100.0)] * dim
        self.name = func_name + '_' + str(dim)
        # CEC 15的 15个function分别是
        # F12015, F22015, F32015, F42015, F52015, F62015, F72015, F82015, F92015, F102015, F112015, F122015, F132015, F142015, F152015
        funcs = opfunu.get_functions_by_classname(func_name)
        self.func = funcs[0](ndim=dim)


    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        X = X.ravel() # shape: [dim]
        if X.shape[0] !=  self.dim:
            assert False
        else:
            fval = self.func.evaluate(X)
        return fval



class GSobol():

    def __init__(self,dim=2):
        self.dim = dim
        self.bounds = [(-5, 5)]*dim
        self.min = 0
        self.fmin = 0
        self.name = 'GSobol'+str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim,lb,ub

    def evaluate(self, X):
        input_dim = self.dim
        a_i = 1

        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:

            gsobol_value = 1
            for i in range(self.dim):
                temp = (np.abs(4*X[...,i]-2)+a_i)/(1+a_i)
                gsobol_value = gsobol_value * temp

        return gsobol_value




class Dixonprice():

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [(-10, 10)] * dim
        # xmin = 2^(-(2^i-2)/2^i)
        # fmin = 0
        self.name = 'Dixonprice' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = self.dim
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            fval = (x1-1)**2
            for i in range(1,self.dim):
                x_i = X[..., i]
                x_i_1 = X[..., i-1]
                fval =fval+ i * (2*x_i**2-x_i_1)**2
        return fval


class StyblinskiTang():

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [(-5, 5)] * dim
        # xmin = [-2.903534]*d
        # fmin = -39.16599*d
        self.name = 'StyblinskiTang' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = self.dim
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            fval = 0
            for i in range(self.dim):
                x_i = X[..., i]
                fval = fval + (x_i**4 - 16*x_i**2 + 5*x_i )
            fval = fval/2
        return fval

class Alpine2():

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = [(0, 10)] * dim
        # xmin = [-7.917]*d
        # fmin = -2.808**d
        self.name = 'StyblinskiTang' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        input_dim =  self.dim
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            fval = 1
            for i in range(self.dim):
                x_i = X[..., i]
                fval = fval * (np.sqrt(x_i) * np.sin(x_i))
            fval = -fval
        return fval



class BraninForrester():

    def __init__(self, dim=2):
        self.dim = 2
        self.bounds = [(-5, 10), (0, 15)]
        # xmin = [-3.689, 13.679]
        # fmin = -16.64402
        self.name = 'BraninForrester' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = 2

        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            fval = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s +5*x1
        return fval

class Cosines():
    def __init__(self, dim=2):
        self.dim = 2
        self.bounds = [(0, 5), (0, 5)]
        # xmin = [0.3125, 0.3125]
        # fmin = -1.6
        self.name = 'Cosines'

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        input_dim = 2
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            g_x1 = (1.6*x1-0.5)**2
            g_x2 = (1.6*x2-0.5)**2
            r_x1 = 0.3*np.cos(3*np.pi*(1.6*x1-0.5))
            r_x2 = 0.3 * np.cos(3 * np.pi * (1.6 * x2 - 0.5))
            fval = -(1 - (g_x1-r_x1)-(g_x2-r_x2))
        return fval

class GoldsteinPrice():

    def __init__(self, dim=2):
        self.dim = 2
        self.bounds = [(-2, 2), (-2, 2)]
        # xmin = [0, -1]
        # fmin = 3
        self.name = 'GoldsteinPrice' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = 2

        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:

            x1 = X[..., 0]
            x2 = X[..., 1]
            part1 = (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
            part2 = (30 + (2 * x1 - 3 * x2) ** 2 * (
                        18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
            fval = part1 * part2
        return fval

class SixHumpCamel():

    def __init__(self, dim=2):
        self.dim = 2
        self.bounds = [(-3, 3), (-2, 2)]
        # xmin = [[0.0898, -0.7126],[-0.0898,0.7126]]
        # fmin = -1.0316
        self.name = 'GoldsteinPrice' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = 2
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            fval = (4-2.1*x1**2 + x1**4 /3)*x1**2 +x1*x2+(-4 + 4*x2**2)*x2**2
        return fval


# Branin objective function
class Branin():
    def __init__(self, dim=2):
        self.dim = 2
        self.bounds =[(-5, 10), (0, 15)]
        # xmin = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
        # fmin = 0.397887
        self.name = 'Branin' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = 2
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        sd = 0

        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[..., 0]
            x2 = X[..., 1]
            fval = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return fval


# Alpine1 objective function
class Alpine1():

    def __init__(self, dim=2):
        self.dim = 5
        self.bounds =[(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
        # xmin = [(0, 0, 0, 0, 0)]
        # fmin = 0
        self.name = 'Alpine1' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        input_dim = 5
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = np.abs(X * np.sin(X) + 0.1 * X).sum()
        return fval


# Egg Holder objective function
class Eggholder():

    def __init__(self, dim=2):
        self.dim = 2
        self.bounds =[(-512, 512), (-512, 512)]
        # xmin = [(512, 404.2319)]
        # fmin = -959.6407
        self.name = 'Eggholder' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub


    def evaluate(self, X):
        input_dim = 2
        X = X.ravel()
        assert X.shape[0] == input_dim
        x1 = X[..., 0]
        x2 = X[..., 1]
        fval = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + 0.5 * x1 + 47))) - x1 * np.sin(
            np.sqrt(np.abs(x1 - (x2 + 47))))
        return fval


class Hartmann6():

    def __init__(self, dim=2):
        self.dim = 6
        self.bounds =[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        # xmin = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        # fmin = -3.32237
        self.name = 'Hartmann6'

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        input_dim = 6
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = np.array([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            alp = np.array([1.0, 1.2, 3.0, 3.2])
            inner_sum = np.sum(A * (X - P) ** 2, axis=-1)
            fval = -(np.sum(alp * np.exp(-inner_sum), axis=-1))
        return fval



class Hartmann3():

    def __init__(self, dim=2):
        self.dim = 3
        self.bounds =[(0, 1), (0, 1), (0, 1)]
        # xmin = (0.114614, 0.555649, 0.852547,)
        # fmin = -3.86278
        self.name = 'Hartmann3'

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        input_dim = 3
        A = np.array([[3,10,30],
                      [0.1,10,35],
                      [3, 10, 30],
                      [0.1, 10, 35]])
        P = np.array([
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.381,  0.5743, 0.8828]])
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            alp = np.array([1.0, 1.2, 3.0, 3.2])
            inner_sum = np.sum(A * (X - P) ** 2, axis=-1)
            fval = -(np.sum(alp * np.exp(-inner_sum), axis=-1))
        return fval




class Ackley():

    def __init__(self, dim=2):
        self.dim = dim
        self.bounds =[(-30, 30)]*self.dim
        # xmin = [(0)]*self.dim
        # fmin = 0
        self.name = 'Ackley' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        input_dim = self.dim
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = -20 * np.exp(-0.2 * np.sqrt((np.sum(X ** 2))/input_dim)) - np.exp(
                np.sum(np.cos(2 * np.pi * X))/ input_dim)+ 20 + np.exp(1)
        return fval

class Rosenbrock():
    def __init__(self, dim=2):
        self.dim = dim
        self.bounds =[(-5, 10)]*self.dim
        # xmin = [(1)]*self.dim
        # fmin = 0
        self.name = 'Rosenbrock' + str(self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub



    def evaluate(self, X):
        input_dim =  self.dim
        X = X.ravel()
        if X.shape[0] != input_dim:
            return 'Wrong input dimension'
        else:
            X = X.reshape(input_dim)
            fval = 0
            for i in range(input_dim-1):
                fval = fval + 100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2
        return fval
