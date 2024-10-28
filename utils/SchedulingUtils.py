import numpy as np

def makespan(times, assignment):
    assigned_times = times * assignment
    row_sums = np.sum(assigned_times, axis=1)
    return np.max(row_sums)