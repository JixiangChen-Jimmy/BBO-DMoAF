from GP_module import GP
import numpy as np
from platypus import NSGAII, Problem, Real, Solution, InjectedPopulation, Archive
from smt.sampling_methods import LHS
from cvxopt import solvers, matrix
from itertools import combinations
from sklearn.cluster import KMeans

class DMEA_tryy:
    def __init__(self, f, lb, ub, num_init, max_iter, k, path, alpha = 0.1, xita = 0.1, lamda = 1, mo_eval=25000, M = 2, func_name=None, random_state = 0, pop_size= 50):
        """
        f: the objective function:
            input: x \in \mathbb{R}^d
            output: scalar value
        lb: lower bound
        ub: upper bound
        num_init: number of initial random sampling
        max_iter: number of iterations
        k: batch size, the total number of function evaluations would be num_init + k * max_iter
        mo_eval: Max number of evaluations for mNSGA-II when optimizing 8-objective optimization
        func_name: Test function name, used in saving path name
        """
        self.f = f
        self.lb = lb.reshape(lb.size)
        self.ub = ub.reshape(ub.size)
        self.dim = self.lb.size
        self.num_init = num_init
        self.max_iter = max_iter
        self.func_name = func_name
        self.path = path
        self.k = k
        self.M = M # Number of acquisition functions that recommend solutions via multi-objectives ensemble
        self.mo_eval = mo_eval # Total function evaluation number for one MOEA evaluation
        self.alpha = alpha # Control parameter in mNSGA-II
        self.xita = xita # Control parameter in NHLA
        self.lamda = lamda # Control parameter in NHLA
        self.random_state = random_state # Random state for initial sampling
        self.pop_size = pop_size  # Population size for MOEA

        self.log_chosen_objectives = None


    def init(self):
            # self.dbx: the previously selected X excepted ones recommended in last iteration
            # self.dby: the previously evaluated results Y excepted ones recommended in last iteration
            # self.lsx: the recommended solutions in last iteration
            # self.lsy: the evaluation results of recommended solutions in last iteration
            # self.best_y: the best evaluation result so far
            self.dbx = np.zeros((self.num_init, self.dim))
            self.dby = np.zeros((self.num_init, 1))
            self.lsx = np.zeros((self.k, self.dim))
            self.lsy = np.zeros((self.k, 1))
            self.best_y = np.inf

            # Latin hypercube sampling
            xlimits = np.array([(self.lb[i], self.ub[i]) for i in range(self.dim)])
            samples = LHS(xlimits=xlimits)
            self.dbx = samples(self.num_init)

            for i in range(self.num_init):
                y = self.f(self.dbx[i])
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = self.dbx[i]
                self.dby[i] = y

            # In the beginning, randomly choose k solutions in initial sampling as the solutions recommended in last iteration
            idxs = np.arange(np.shape(self.dbx)[0])
            idxs = np.random.permutation(idxs)
            if self.num_init <= self.k:
                num = int(0.5 * self.num_init)
            else:
                num = self.k
            idxs = idxs[0: num]

            self.lsx = self.dbx[idxs]
            self.lsy = self.dby[idxs]
            self.dbx = np.delete(self.dbx, idxs, 0)
            self.dby = np.delete(self.dby, idxs, 0)

            self.last_best_y = np.min(self.dby)
            self.log_best_y = np.array([self.best_y])
            # Construct the initial GP model
            self.model = GP(train_x=self.dbx, train_y=self.dby, lb=self.lb, ub=self.ub, k=self.k, num_init= self.num_init)

    def NLHA(self):
        # Offline version of NLHA
        # @article{10.1162/evco_a_00223,
        #     author = {Li, Yifan and Liu, Hai-Lin and Goodman, E. D.},
        #     title = "{Hyperplane-Approximation-Based Method for Many-Objective Optimization Problems with Redundant Objectives}",
        #     journal = {Evolutionary Computation},
        #     volume = {27},
        #     number = {2},
        #     pages = {313-344},
        #     year = {2019},
        #     month = {06},
        #     doi = {10.1162/evco_a_00223},
        #     url = {https://doi.org/10.1162/evco\_a\_00223},
        # }
        # Magnitude adjustment of each objective
        F = np.zeros(np.shape(self.pf))
        for i in range(np.shape(F)[1]):
            F[:, i] = (self.pf[:, i] - np.min(self.pf[:, i])) / (np.max(self.pf[:, i]) - np.min(self.pf[:, i]))
        # Choose a proper constant $q$ for power transformation
        q_options = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
        min_error = np.inf
        w_optimal = []
        # Implement cvxopt to solve the constrained quadratic programming problem, the best choice of $q$ is with the smallest error
        for q_power in q_options:
            F_q = np.power(F,q_power)
            P = matrix(F_q.T @ F_q)
            q = matrix((-np.ones((1,np.shape(F)[0]))@F_q + self.lamda/2*np.ones((1,np.shape(F)[1]))).T)
            G = matrix(-np.eye(np.shape(F)[1]))
            h = matrix(np.zeros((np.shape(F)[1], 1)))
            solvers.options['show_progress'] = False
            tol = 1e-5
            solvers.options['feastol'] = tol
            sol = solvers.qp(P,q,G,h)
            w = sol['x']
            error = sol['primal objective']
            if error < min_error:
                w_optimal = w
                min_error = error
        w_optimal = np.array(w_optimal).ravel()
        max_w = np.max(w_optimal)
        obj = []
        # Identify the essential objectives
        for i in range(len(w_optimal)):
            if w_optimal[i] > self.xita * max_w:
                obj.append(i)
        min_correlation_coefficient = np.inf
        min_idx = -1
        # In case of reduction to only one objective, select another one with the smallest correlation coefficient
        if len(obj) == 1:
            for i in range(np.shape(F)[1]):
                if i == obj[0]:
                    continue
                correlation_coefficient = np.corrcoef(np.concatenate((np.reshape(self.pf[:, obj[0]], (1,-1)), np.reshape(self.pf[:, i], (1,-1))), 0))[0, 1]
                if correlation_coefficient < min_correlation_coefficient:
                    min_correlation_coefficient = correlation_coefficient
                    min_idx = i
            obj.append(min_idx)
        return obj

    def optimize(self):
        for iter in range(self.max_iter):

            def obj_mNSGA_II_8(x):
                # Modify each objective to transform NSGA-II into mNSGA-II
                ei_0, ei_1, ei_2, pi_0, pi_1, pi_2, gp_lcb, mes = self.model.MACE_8(np.array([x]))
                # acq_val = [-ei_0, -ei_1, -ei_2, -pi_0, -pi_1, -pi_2, gp_lcb, -mes]
                acq_val = [gp_lcb, -ei_0, -pi_0, -ei_1, -pi_1, -ei_2, -pi_2, -mes]
                mean_val = np.mean(acq_val)
                f = [(1-self.alpha)*fi + self.alpha*mean_val for fi in acq_val]
                return f

            def obj_NSGA_II_8(x):
                ei_0, ei_1, ei_2, pi_0, pi_1, pi_2, gp_lcb, mes = self.model.MACE_8(np.array([x]))
                # acq_val = [-ei_0, -ei_1, -ei_2, -pi_0, -pi_1, -pi_2, gp_lcb, -mes]
                acq_val = [gp_lcb, -ei_0, -pi_0, -ei_1, -pi_1, -ei_2, -pi_2, -mes]
                return acq_val

            def obj_essential_NSGA_II(x):
                return np.array(obj_NSGA_II_8(x))[self.essential_combination_idx].tolist()

            def obj_essential_mNSGA_II(x):
                return np.array(obj_mNSGA_II_8(x))[self.essential_combination_idx].tolist()


            def obj_comb(x):
                # During enumeration of the 2-acquisition function combination
                return np.array(obj_NSGA_II_8(x))[self.current_combination_idx].tolist()

            def obj_best(x):
                # Use the chosen acquisition functions as objectives to recommend solutions in this iteration
                return np.array(obj_NSGA_II_8(x))[self.best_combination_idx].tolist()

            arch = Archive() # Stores solutions that belong to approximate Pareto Set among all evaluations of mNSGA-II
            problem = Problem(self.dim, 8)
            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])
            problem.function = obj_mNSGA_II_8

            if iter > 0:
                # In each iteration, the initial population for mNSGA-II is partly inherited from previous Pareto Set
                num_init = max(min(int(0.6 * np.shape(self.ps)[0]), 10), 1)
                idxs = np.arange(np.shape(self.ps)[0])
                idxs = np.random.permutation(idxs)
                idxs = idxs[0: num_init]
                init_s = [Solution(problem) for _ in range(num_init)]
                for i in range(num_init):
                    init_s[i].variables = [x for x in self.ps[idxs[i], :]]
                gen = InjectedPopulation(init_s)
                algorithm = NSGAII(problem, population=self.pop_size, generator=gen, archive=arch)
            else:
                # In the beginning, the population for mNSGA-II are totally random
                algorithm = NSGAII(problem, population=self.pop_size, archive=arch)


            algorithm.run(self.mo_eval)

            if len(algorithm.result) > self.k:
                optimized = algorithm.result
            else:
                optimized = algorithm.population

            self.pf = np.array([s.objectives for s in optimized])
            self.ps = np.array([s.variables for s in optimized])
            self.pf_8d = self.pf.copy()
            self.ps_8d = self.ps.copy()

            u_ps, u_idx = np.unique(self.ps_8d, return_index= True, axis= 0)
            u_idx = u_idx.tolist()
            self.pf_8d = self.pf_8d[u_idx]
            self.ps_8d = self.ps_8d[u_idx]

            # Remove identical solutions in PS
            # self.unique()
            # print(f'Before unique, ps shape: {np.shape(self.ps)}')
            u_ps, u_idx = np.unique(self.ps, return_index= True, axis= 0)
            u_idx = u_idx.tolist()
            self.pf = self.pf[u_idx]
            self.ps = self.ps[u_idx]
            # print(f'After unique, ps shape: {np.shape(self.ps)}')
            try:
                # Get the essential objectives of original 10-d MaOP
                self.essential_combination_idx = self.NLHA()
            except ValueError:
                self.essential_combination_idx = [0,1,2,3,4,5,6,7]
            # print(f'Essential objectives: {self.essential_combination_idx}')
            best_quality = -np.inf
            self.best_combination_idx = None
            # The initial population of mNSGA-II for the 2-d MOP is partly inherited from the previous 10-d PS
            num_inherit = max(min(int(0.6 * np.shape(self.ps_8d)[0]), 10), 1)
            idxs = np.arange(np.shape(self.ps_8d)[0])
            idxs = np.random.permutation(idxs)
            idxs = idxs[0: num_inherit]

            for combination in combinations(self.essential_combination_idx, self.M):
                # Enumerate the 2-objective combination among essential objectives(acquisition functions)
                # Judge each combination by exploiting information of solutions recommended in last iteration
                quality = 0
                obj_idx = np.array(combination)
                self.current_combination_idx = obj_idx

                # Finetune the PS of 2-d MOP
                arch = Archive()
                problem = Problem(self.dim, self.M)
                for i in range(self.dim):
                    problem.types[i] = Real(self.lb[i], self.ub[i])
                problem.function = obj_comb
                init_s = [Solution(problem) for _ in range(num_inherit)]
                for i in range(num_inherit):
                    init_s[i].variables = [x for x in self.ps_8d[idxs[i], :]]
                gen = InjectedPopulation(init_s)
                algorithm = NSGAII(problem, population=self.pop_size, generator=gen, archive=arch)
                algorithm.run(self.mo_eval)
                if len(algorithm.result) > self.k:
                    optimized = algorithm.result
                else:
                    optimized = algorithm.population

                pf_M_obj = np.array([s.objectives for s in optimized])
                ps_M_obj = np.array([s.variables for s in optimized])

                u_ps, u_idx = np.unique(ps_M_obj, return_index=True, axis=0)
                u_idx = u_idx.tolist()
                pf_M_obj = pf_M_obj[u_idx]
                ps_M_obj = ps_M_obj[u_idx]

                # Calculate the quality of each combination, choose the one with highest quality
                for i in range(np.shape(self.lsy)[0]):
                    pf_lsx = np.array(self.model.MACE_8(self.lsx[i]))[obj_idx]
                    dis_list = [(pf_M_obj[j] - pf_lsx)@(pf_M_obj[j] - pf_lsx).T for j in range(np.shape(pf_M_obj)[0])]
                    dis = np.min(dis_list)
                    quality += (self.last_best_y - self.lsy[i])/(dis + 1)
                if quality > best_quality:
                    best_quality = quality
                    self.best_combination_idx = obj_idx

            # Update the dataset and train the new GP model used to recommend solutions in this iteration
            self.dbx = np.concatenate((self.dbx, self.lsx), axis= 0)
            self.dby = np.concatenate((self.dby, self.lsy), axis= 0)
            self.last_best_y = self.best_y
            self.model.update_model(self.dbx, self.dby)

            arch = Archive()
            problem = Problem(self.dim, self.M)
            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])
            problem.function = obj_best
            # The initial population of mNSGA-II for the 2-d MOP is partly inherited from the previous chosen 2-d PS
            num_inherit = max(min(int(0.6 * np.shape(self.ps)[0]), 10), 1)
            idxs = np.arange(np.shape(self.ps)[0])
            idxs = np.random.permutation(idxs)
            idxs = idxs[0: num_inherit]
            init_s = [Solution(problem) for _ in range(num_inherit)]
            for i in range(num_inherit):
                init_s[i].variables = [x for x in self.ps[idxs[i], :]]
            gen = InjectedPopulation(init_s)
            algorithm = NSGAII(problem, population=self.pop_size, generator=gen, archive=arch)
            algorithm.run(self.mo_eval)

            if len(algorithm.result) > self.k:
                optimized = algorithm.result
            else:
                optimized = algorithm.population

            self.pf = np.array([s.objectives for s in optimized])
            self.ps = np.array([s.variables for s in optimized])
            u_ps, u_idx = np.unique(self.ps, return_index= True, axis= 0)
            u_idx = u_idx.tolist()
            self.pf = self.pf[u_idx]
            self.ps = self.ps[u_idx]

            if np.shape(self.pf)[0] < self.k:
                # In some cases, it turns out the size of pareto set given by optimizing the 2d-MOP is smaller than the batch size
                # For this situation, we use the essential objectives to form the MOP and get corresponding pareto set
                print(f'{self.func_name}, batch_size: {self.k}, id: {self.random_state}, iter: {iter}, fall into essential objectives because size of {np.shape(self.pf)}')
                arch = Archive()
                problem = Problem(self.dim, len(self.essential_combination_idx))
                for i in range(self.dim):
                    problem.types[i] = Real(self.lb[i], self.ub[i])
                if len(self.essential_combination_idx) > 3:
                    problem.function = obj_essential_mNSGA_II
                else:
                    problem.function = obj_essential_NSGA_II
                # The initial population is partly inherited from 8-d PS
                num_inherit = max(min(int(0.6 * np.shape(self.ps_8d)[0]), 10), 1)
                idxs = np.arange(np.shape(self.ps_8d)[0])
                idxs = np.random.permutation(idxs)
                idxs = idxs[0: num_inherit]
                init_s = [Solution(problem) for _ in range(num_inherit)]
                for i in range(num_inherit):
                    init_s[i].variables = [x for x in self.ps_8d[idxs[i], :]]
                gen = InjectedPopulation(init_s)
                algorithm = NSGAII(problem, population=self.pop_size, generator=gen, archive=arch)
                algorithm.run(self.mo_eval)

                if len(algorithm.result) > self.k:
                    optimized = algorithm.result
                else:
                    optimized = algorithm.population

                self.pf = np.array([s.objectives for s in optimized])
                self.ps = np.array([s.variables for s in optimized])
                u_ps, u_idx = np.unique(self.ps, return_index=True, index=0)
                u_idx = u_idx.tolist()
                self.pf = self.pf[u_idx]
                self.ps = self.ps[u_idx]
                # In some cases, even essential objectives may fail to get enough choice in pareto set.
                # This situation occurs when the essential objectives are exactly the previously used 2-d MOP.
                # And we will use the pareto set of the 8d MaOP.
                if np.shape(self.ps)[0] < self.k:
                    self.pf = self.pf_8d
                    self.ps = self.ps_8d
                    print(f'{self.func_name}, batch_size: {self.k}, id: {self.random_state}, iter: {iter}, fall into 8d because size of essential pf: {np.shape(self.pf)}, size of 8d pf: {np.shape(self.pf_8d)}')
                else:
                    print(f'{self.func_name}, batch_size: {self.k}, id: {self.random_state}, iter: {iter}, size of essential pf: {np.shape(self.pf)}, size of 8d pf: {np.shape(self.pf_8d)}')
            # Choose extreme solutions of the two objective
            solutions_idx = []
            for i in range(np.shape(self.pf)[1]):
                solutions_idx.append(np.argmin(self.pf[:, i]))

            solutions_idx = list(set(solutions_idx))
            if len(solutions_idx) > self.k:
                solutions_idx = solutions_idx[:self.k]
            self.lsx = self.ps[solutions_idx]
            num_cluster = self.k - np.shape(self.lsx)[0]
            if num_cluster > 0:
                self.ps = np.delete(self.ps, solutions_idx, 0)
                self.pf = np.delete(self.pf, solutions_idx, 0)
                # Cluster the remaining solutions in PS, choose the solution in each cluster with the smallest posterior mean value
                cluster_idx = KMeans(n_clusters= num_cluster).fit_predict(self.pf)
                valid_cluster_idx = list(set(cluster_idx))
                if len(valid_cluster_idx) < num_cluster:
                    cluster_idx = KMeans(n_clusters=num_cluster).fit_predict(self.ps)
                for i in range(num_cluster):
                    cluster_x = self.ps[np.where(cluster_idx == i)]
                    idx = np.argmin([self.model.predict(x)[0] for x in cluster_x])
                    self.lsx = np.concatenate((self.lsx, np.reshape(cluster_x[idx], (1,-1))), axis= 0)

            self.lsy = []
            for x in self.lsx:
                y = self.f(x)
                if len(self.lsy) == 0:
                    self.lsy = np.array([y]).reshape((-1,1))
                else:
                    self.lsy = np.concatenate((self.lsy, np.array([y]).reshape((-1,1))), axis= 0)
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x
            # Save results
            self.log_best_y = np.concatenate((self.log_best_y, np.array([self.best_y])))
            np.savetxt(f'{self.path[1]}/dbx_Batchsize_{self.k}_{self.func_name}_{self.random_state}.txt', np.concatenate((self.dbx, self.lsx), axis= 0))
            np.savetxt(f'{self.path[1]}/dby_Batchsize_{self.k}_{self.func_name}_{self.random_state}.txt', np.concatenate((self.dby, self.lsy), axis= 0))
            np.savetxt(f'{self.path[0]}/besty_Batchsize_{self.k}_{self.func_name}_{self.random_state}.txt', self.log_best_y)

            if self.log_chosen_objectives is None:
                self.log_chosen_objectives = np.array([self.best_combination_idx]).reshape((1,-1))
            else:
                self.log_chosen_objectives = np.concatenate((self.log_chosen_objectives, np.array([self.best_combination_idx]).reshape((1,-1))), axis= 0)
            np.savetxt(f'{self.path[2]}/chosen_objectives_Batchsize_{self.k}_{self.func_name}_{self.random_state}.txt', self.log_chosen_objectives, fmt = '%d')