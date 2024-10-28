import numpy as np
import matplotlib.pyplot as plt

def plot_ratios(arr):
    opt = arr[:,0]
    ssg = arr[:,1]
    sg = arr[:,2]
    asg = arr[:,3] 
    err = arr[:,4]
    eta = arr[:,5]

    # Plot the lines
    plt.scatter(err, opt/opt, label='opt', color='red', marker='x')
    plt.scatter(err, asg/opt, label='asg', color='green')
    plt.scatter(err, ssg/opt, label='ssg', color='blue')
    plt.scatter(err, sg/opt, label='sg', color='purple', marker='*')
    plt.legend()

    # Add title and axis labels
    plt.title('Mechanism ratio as a function of err')
    plt.xlabel('error rho hat')
    plt.ylabel('competitive ratio')
    plt.show()

def plot_errors(arr):
    opt = arr[:,0]
    ssg = arr[:,1]
    sg = arr[:,2]
    asg = arr[:,3] 
    err = arr[:,4]
    eta = arr[:,5]

    _, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(sg, err, color='blue')
    ax1.set_title('rho hat error vs SG')
    ax1.set_xlabel('Scaled Greedy ratio')
    ax1.set_ylabel("eta error")

    ax2.scatter(sg, eta, color='red')
    ax2.set_title('eta error vs SG')
    ax2.set_xlabel('Scaled Greedy ratio')
    ax2.set_ylabel("eta error")

    plt.title('Errors as a function of ScaledGreedy performance')
    plt.tight_layout()
    plt.show()



#opt, ssg, sg, asg, err, eta
data= np.load('experiments/Sched/preds100.npz')
arr = data['array1']

plot_ratios(arr)
plot_errors(arr)