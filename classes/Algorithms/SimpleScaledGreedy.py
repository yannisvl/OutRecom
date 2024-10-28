import numpy as np

def argmin_with_tie_breaking(column, tie):
    min_index = np.argmin(column)
    min_value = column[min_index]
    
    tie_indices = np.where(column == min_value)[0]
    
    for x in tie_indices:
      if x==tie:
        return x

    return tie_indices[0]

def vcg(a, ties):
    (rows, cols) = a.shape
    result_array = np.zeros((rows, cols))
    argmin_indices = [argmin_with_tie_breaking(a[:, j], ties[j]) for j in range(cols)]
    result_array[argmin_indices, np.arange(cols)] = 1
    return result_array

class SimpleScaledGreedy():
    def __init__(self):
        pass

    def solve(self, actual, pred, pred_opt):
        n, _ = pred.shape
        ties = np.argmax(pred_opt, axis=0)

        #p i_j hat
        pred_opt_values2d = pred_opt * pred
        pred_opt_values = np.amax(pred_opt_values2d, axis=0)

        #find weights
        weights = np.clip(np.tile(pred_opt_values, (pred.shape[0], 1)) / pred, 1, n)
        weighted = weights * actual

        return vcg(weighted, ties)