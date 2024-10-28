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

class AllocationScaledGreedy():
    def __init__(self, beta):
        self.b = beta

    def solve(self, actual, pred):
        n = pred.shape[0]
        ties = np.argmax(pred, axis=0)
        weights = np.where(pred > 0, 1, n / self.b)

        weighted = weights * actual

        return vcg(weighted, ties)