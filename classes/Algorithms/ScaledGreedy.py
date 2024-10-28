import numpy as np

def argmin_with_tie_breaking(column, tie_priority):
    min_index = np.argmin(column)
    min_value = column[min_index]
    tie_indices = np.where(column == min_value)[0]
    
    index_where_1 = np.where(tie_priority == 1)[0][0]
    if index_where_1 in tie_indices:
        return index_where_1

    indices_where_2 = np.where(tie_priority == 2)[0]
    for x in indices_where_2:
      if x in tie_indices:
          return x

    return tie_indices[0]

def vcg(a, ties):
    (rows, cols) = a.shape
    result_array = np.zeros((rows, cols))
    argmin_indices = [argmin_with_tie_breaking(a[:, j], ties[:, j]) for j in range(cols)]
    result_array[argmin_indices, np.arange(cols)] = 1
    return result_array

def fill_T(pred, pred_opt_values, I, Jobs):
    T = set()
    min_values = np.min(pred, axis=0)
    for col_index, min_value in enumerate(min_values):
        # Find all occurrences of the minimum value in the column
        min_indices = np.where(pred[:, col_index] == min_value)[0]
        for x in min_indices:
            if x in I and col_index in Jobs and pred[x,col_index] < pred_opt_values[col_index]:
                T.add((x, col_index))
    return T

def get_max_pair_T(a, b, T):
    max_ratio = None
    max_ratio_pair = None
    for i, j in T:
        ratio = b[j] / a[i,j]
        if max_ratio is None or ratio > max_ratio:
            max_ratio = ratio
            max_ratio_pair = (i, j)
    
    return max_ratio_pair

class ScaledGreedy():
    def __init__(self, gamma):
        self.gamma = gamma

    def solve(self, actual, pred, pred_opt, pred_assign_makespan):
        (n, m) = pred.shape
        ties = pred_opt.copy()

        #p i_j hat
        pred_opt_values2d = pred_opt * pred
        pred_opt_values = np.amax(pred_opt_values2d, axis=0)

        #find weights
        weights = np.maximum(1, np.minimum(np.tile(pred_opt_values, (pred.shape[0], 1)) / pred, n))
        J = {i: set() for i in range(n)}
        I = set(range(n))
        T = fill_T(pred, pred_opt_values, I, range(m))

        while T:
            i_star, j_star = get_max_pair_T(pred, pred_opt_values, T)
            J[i_star].add(j_star)
            for i in range(n):
                if pred[i_star, j_star] <= pred[i, j_star]:
                    weights[i, j_star] = 1
            I = set()
            for i in range(n):
                s = sum([pred[i, j] for j in J[i]])
                if s < self.gamma*pred_assign_makespan:
                    I.add(i)

            jobs = set(range(m)).difference(set().union(*J.values()))
            T = fill_T(pred, pred_opt_values, I, jobs)

        #run vcg
        weighted = weights * actual
        for row in J.keys():
            for col in J[row]:
                ties[row, col] = 2

        return vcg(weighted, ties)