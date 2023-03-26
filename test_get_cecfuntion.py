import opfunu
import numpy as np

class F1():

    def __init__(self, dim=10):
        self.dim = dim
        self.bounds = [(-100.0, 100.0)] * dim
        self.name = 'F1_' + str(self.dim)

        self.funcs = opfunu.get_functions_by_classname("F12015")
        self.func = self.funcs[0](ndim=self.dim)

    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        X = X.ravel()
        # shape: [dim]
        if X.shape[0] !=  self.dim:
            assert False
        else:
            fval = self.func.evaluate(X)
        return fval

# X = [20]*10

# fun_n = F1(dim=10)

# funcs = opfunu.get_functions_by_classname("F12015")
# func = funcs[0](ndim=10)
# func.evaluate(func.create_solution())

## or
# 15个function分别是 F12015, F22015, F32015, F42015, F52015, F62015, F72015, F82015, F92015, F102015, F112015, F122015, F132015, F142015, F152015
class F3():
    def __init__(self, dim=10):
        self.dim = dim
        self.bounds = [(-100.0, 100.0)] * dim
        self.name = 'F3_' + str(self.dim)
        self.funcs = opfunu.get_functions_by_classname("F32015")
        self.func = self.funcs[0](ndim=self.dim)

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

from opfunu.cec_based.cec2015 import F22015

X = [20]*10
funcs = opfunu.get_functions_by_classname("F22015")
print(funcs)
func = funcs[0](ndim=10)
fval = func.evaluate(X)
print(fval)

func_2 = F22015(ndim=10)
fval_2 = func_2.evaluate(X)
print(fval_2)