from utils.SchedulingUtils import makespan

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import gurobipy as gp
from gurobipy import GRB

def greedy_assignment(p):
    (rows, cols) = p.shape
    result_array = np.zeros((rows, cols))
    argmin_indices = np.argmin(p, axis=0)
    result_array[argmin_indices, np.arange(cols)] = 1
    return result_array

def LP(p, C):
    (n, m) = p.shape
    l = n * m

    S = np.where(p <= C, 1, 0)

    # Create a new Gurobi model
    model = gp.Model()

    # Set Gurobi parameters to disable logging
    model.setParam('OutputFlag', 0)
    model.setParam('LogToConsole', 0)

    # Create variables
    x = model.addVars(l, vtype=GRB.CONTINUOUS, name="x")

    # Set objective function (minimize sum of variables)
    model.setObjective(0, GRB.MINIMIZE)

    # Add constraints
    for j in range(m):
        model.addConstr(gp.quicksum(x[i * m + j] for i in range(n) if S[i, j] > 0) == 1, f"eq_{j}")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i * m + j] * p[i, j] for j in range(m) if S[i, j] > 0) <= C, f"ub_{i}")

    for i in range(n):
        for j in range(m):
            if S[i,j]>0:
                model.addConstr(x[i * m + j] >= 0, f"lb_{i}")

    # Optimize the model
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return False, []

    # Retrieve solution
    solution = np.zeros(l)
    for i in range(l):
        solution[i] = x[i].X

    return True, solution

class OptScheduling2():
    def __init__(self, iterations):
        self.binary_search_iterations = iterations

    def binary_search(self, p):
        a_greedy = greedy_assignment(p)
        a = makespan(p, a_greedy)
        minval = a / p.shape[0]
        maxval = a
        i=0
        success = False
        while not success or i<self.binary_search_iterations:
            i+=1
            val = (minval + maxval) / 2
            success, x_sol = LP(p, val)
            if success:
                maxval = val
                last_success_x = x_sol
            else:
                minval = val

        return last_success_x.reshape(p.shape)

    #supm - rounding
    def solve(self, p):
        sol = np.zeros_like(p)

        (n, m) = p.shape
        x_vertex = self.binary_search(p)

        jobs_given = set()
        for i in range(n):
            for j in range(m):
                if x_vertex[i,j]>=1:
                    sol[i, j] = 1
                    jobs_given.add(j)
       
        h = np.where((x_vertex > 0) & (x_vertex < 1), 1, 0)
        h[:, list(jobs_given)] = 0
        
        bip_graph = csr_matrix(h)
        match = maximum_bipartite_matching(bip_graph, perm_type='column')
        for i, j in enumerate(match):
            if j > -1:
                sol[i, j] = 1

        return sol