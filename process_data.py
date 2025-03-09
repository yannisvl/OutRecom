import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def compare_errors(arr, file):
    sorted_indices = np.argsort(arr[:, 0])
    sorted_array = arr[sorted_indices]
    ratios = sorted_array[:, 0]
    ratio_err = sorted_array[:, 1]
    dist_err = sorted_array[:, 3]

    plt.figure()
    plt.scatter(ratios, ratio_err, color='blue', label=r"$\hat{\rho}$")
    plt.scatter(ratios, dist_err, color='red', label=r"$\eta$")

    dataset, rest = file.split('c')
    c = rest.split('preds')[0]
    plt.title(dataset + ", c = " + c)
    plt.xlabel('CMP ratio')
    plt.ylabel('Error')

    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/"+file+".png")


figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
for file in os.listdir("experiments"):    
    data= np.load("experiments/"+file)
    arr = data['array1']
    compare_errors(arr, file)