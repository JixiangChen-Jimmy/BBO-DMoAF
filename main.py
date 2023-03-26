import numpy as np

from DMEA_method import DMEA_tryy
import objective
import os
import threading
from joblib import Parallel, delayed
np.set_printoptions(precision=12, linewidth=500)

def get_func(function_name):
    f_n_set_dim={}
    # 2d
    f_n_set_dim['EggHolder'] = objective.Eggholder()
    f_n_set_dim['Branin'] = objective.Branin()
    f_n_set_dim['Rosenbrock2'] = objective.Rosenbrock(dim=2)
    f_n_set_dim['Ackley2'] = objective.Ackley(dim=2)
    f_n_set_dim['Cosines'] = objective.Cosines()
    f_n_set_dim['SixHumpCamel'] = objective.SixHumpCamel()
    f_n_set_dim['GoldsteinPrice'] = objective.GoldsteinPrice()
    f_n_set_dim['GSobol2'] = objective.GSobol(dim=2)
    f_n_set_dim['StyblinskiTang2'] = objective.StyblinskiTang(dim=2)
    # 3d
    f_n_set_dim['Hartman3'] = objective.Hartmann3()
    f_n_set_dim['GSobol3'] = objective.GSobol(dim=3)
    f_n_set_dim['Ackley3'] = objective.Ackley(dim=3)
    f_n_set_dim['StyblinskiTang3'] = objective.StyblinskiTang(dim=3)
    # 5d
    f_n_set_dim['Ackley5'] = objective.Ackley(dim=5)
    f_n_set_dim['GSobol5'] = objective.GSobol(dim=5)
    f_n_set_dim['Alpine2_5'] = objective.Alpine2(dim=5)
    f_n_set_dim['Alpine1'] = objective.Alpine1()
    f_n_set_dim['StyblinskiTang5'] = objective.StyblinskiTang(dim=5)
    # 6d
    f_n_set_dim['Hartman6'] = objective.Hartmann6()
    f_n_set_dim['GSobol6'] = objective.GSobol(dim=6)
    f_n_set_dim['Alpine2_6'] = objective.Alpine2(dim=6)
    f_n_set_dim['Ackley6'] = objective.Ackley(dim=6)
    f_n_set_dim['StyblinskiTang6'] = objective.StyblinskiTang(dim=6)
    # 10d
    f_n_set_dim['GSobol10'] = objective.GSobol(dim=10)
    f_n_set_dim['Ackley10'] = objective.Ackley(dim=10)
    f_n_set_dim['Rosenbrock10'] = objective.Rosenbrock(dim=10)
    f_n_set_dim['Dixonprice10'] = objective.Dixonprice(dim=10)
    f_n_set_dim['StyblinskiTang10'] = objective.StyblinskiTang(dim=10)
    f_n_set_dim['Alpine2_10'] = objective.Alpine2(dim=10)
    #20d
    f_n_set_dim['GSobol20'] = objective.GSobol(dim=20)
    f_n_set_dim['Ackley20'] = objective.Ackley(dim=20)
    f_n_set_dim['Rosenbrock20'] = objective.Rosenbrock(dim=20)
    f_n_set_dim['Dixonprice20'] = objective.Dixonprice(dim=20)
    f_n_set_dim['StyblinskiTang20'] = objective.StyblinskiTang(dim=20)
    # CEC 15, 10d
    f_n_set_dim['F1_10d'] = objective.CEC_2015_functions(dim=10, func_name='F12015')
    f_n_set_dim['F2_10d'] = objective.CEC_2015_functions(dim=10, func_name='F22015')
    f_n_set_dim['F3_10d'] = objective.CEC_2015_functions(dim=10, func_name='F32015')
    f_n_set_dim['F4_10d'] = objective.CEC_2015_functions(dim=10, func_name='F42015')
    f_n_set_dim['F5_10d'] = objective.CEC_2015_functions(dim=10, func_name='F52015')
    f_n_set_dim['F6_10d'] = objective.CEC_2015_functions(dim=10, func_name='F62015')
    f_n_set_dim['F7_10d'] = objective.CEC_2015_functions(dim=10, func_name='F72015')
    f_n_set_dim['F8_10d'] = objective.CEC_2015_functions(dim=10, func_name='F82015')
    f_n_set_dim['F9_10d'] = objective.CEC_2015_functions(dim=10, func_name='F92015')
    f_n_set_dim['F10_10d'] = objective.CEC_2015_functions(dim=10, func_name='F102015')
    f_n_set_dim['F11_10d'] = objective.CEC_2015_functions(dim=10, func_name='F112015')
    f_n_set_dim['F12_10d'] = objective.CEC_2015_functions(dim=10, func_name='F122015')
    f_n_set_dim['F13_10d'] = objective.CEC_2015_functions(dim=10, func_name='F132015')
    f_n_set_dim['F14_10d'] = objective.CEC_2015_functions(dim=10, func_name='F142015')
    f_n_set_dim['F15_10d'] = objective.CEC_2015_functions(dim=10, func_name='F152015')
    # CEC 15, 30d
    f_n_set_dim['F1_30d'] = objective.CEC_2015_functions(dim=30, func_name='F12015')
    f_n_set_dim['F2_30d'] = objective.CEC_2015_functions(dim=30, func_name='F22015')
    f_n_set_dim['F3_30d'] = objective.CEC_2015_functions(dim=30, func_name='F32015')
    f_n_set_dim['F4_30d'] = objective.CEC_2015_functions(dim=30, func_name='F42015')
    f_n_set_dim['F5_30d'] = objective.CEC_2015_functions(dim=30, func_name='F52015')
    f_n_set_dim['F6_30d'] = objective.CEC_2015_functions(dim=30, func_name='F62015')
    f_n_set_dim['F7_30d'] = objective.CEC_2015_functions(dim=30, func_name='F72015')
    f_n_set_dim['F8_30d'] = objective.CEC_2015_functions(dim=30, func_name='F82015')
    f_n_set_dim['F9_30d'] = objective.CEC_2015_functions(dim=30, func_name='F92015')
    f_n_set_dim['F10_30d'] = objective.CEC_2015_functions(dim=30, func_name='F102015')
    f_n_set_dim['F11_30d'] = objective.CEC_2015_functions(dim=30, func_name='F112015')
    f_n_set_dim['F12_30d'] = objective.CEC_2015_functions(dim=30, func_name='F122015')
    f_n_set_dim['F13_30d'] = objective.CEC_2015_functions(dim=30, func_name='F132015')
    f_n_set_dim['F14_30d'] = objective.CEC_2015_functions(dim=30, func_name='F142015')
    f_n_set_dim['F15_30d'] = objective.CEC_2015_functions(dim=30, func_name='F152015')


    return f_n_set_dim[function_name]


def run_benchmark(batch_size, f_n, path):
    fun_n = get_func(f_n)
    dim, lb, ub = fun_n.get_bound()

    num_init = 5 * dim
    if dim==10:
        max_iter = int((500-num_init)/batch_size)
    if dim==30:
        max_iter = int((1500-num_init)/batch_size)
    b = os.getcwd().replace('\\', '/')
    y_best_path = f'y_best/{f_n}/batch{batch_size}'
    for kkk in range(10):
        print(f'{f_n}_batch_{batch_size}_{kkk} start')
        pp = b + '/' + y_best_path + f'/besty_Batchsize_{batch_size}_{f_n}_{kkk}.txt'
        isExist = os.path.exists(pp)
        if isExist:
            result = np.loadtxt(pp)
            if len(result) == max_iter + 1:
                print(f'{f_n}_{batch_size} skip!')
                continue
        optimizer = DMEA_tryy(fun_n.evaluate, lb, ub, num_init, max_iter, batch_size, mo_eval=10000, func_name=f_n, path=path,
                              random_state=kkk, alpha=0.1)
        optimizer.init()
        optimizer.optimize()
    print(f'{f_n}_{batch_size} done!')

if __name__ == '__main__':

    # test function

    total = ['Ackley2', 'Ackley3', 'Ackley5', 'Ackley6', 'Ackley10', 'Ackley20',
             'GSobol2', 'GSobol3', 'GSobol5', 'GSobol6', 'GSobol10', 'GSobol20',
             'StyblinskiTang2', 'StyblinskiTang3', 'StyblinskiTang5', 'StyblinskiTang6', 'StyblinskiTang10',
             'StyblinskiTang20',
             'F1_10d', 'F2_10d', 'F3_10d', 'F4_10d', 'F5_10d', 'F6_10d', 'F7_10d', 'F8_10d', 'F9_10d', 'F10_10d',
             'F11_10d', 'F12_10d', 'F13_10d', 'F14_10d', 'F15_10d'
              'F1_30d', 'F2_30d', 'F3_30d', 'F4_30d', 'F5_30d', 'F6_30d', 'F7_30d', 'F8_30d', 'F9_30d', 'F10_30d',
             'F11_30d', 'F12_30d', 'F13_30d', 'F14_30d', 'F15_30d']



    batch_size = [5,10]
    tasks = []
    for k in batch_size:
        for f_n in total:
            y_best_path = f'y_best/{f_n}/batch{k}'
            dbxy_path = f'dbxy/{f_n}/batch{k}'
            fig_path = f'chosen_combination/{f_n}/batch{k}'
            path = [y_best_path, dbxy_path, fig_path]
            b = os.getcwd().replace('\\', '/')
            for pa in path:
                isExist = os.path.exists(b + '/' + pa)
                if not isExist:
                    os.makedirs(b + '/' + pa)
            tasks.append(delayed(run_benchmark)(k, f_n, path))

    multi_work = Parallel(n_jobs= -1)
    multi_work(tasks)