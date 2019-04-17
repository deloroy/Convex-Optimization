#Convex Optimization -Homework 3
#Author : Yonatan Deloro

import math
import matplotlib.pyplot as plt

from lasso import *
from barrierMethod import *

def main():
    #generating data and building lasso dual problem
    X,Y,w = generate_data(n=50,d=500,sparsity_w=0.8, seed=1)  #seed used for report experiments
    lambd = 10
    [Q,p,A,b] = build_QP_lasso(X,Y,lambd)

    #various values for the barrier method parameter
    mus = [2,5,10,15,20,50,1000]
    #for plotting
    colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']

    gaps = {} #will store the sequence of gaps f(vt) -f* for each mu
    # (the gap is computed at the end of each centering step)
    # (to plot step functions, I hence repeated n times the final computation if there was n inner Newton iterations)

    # solving the QP problem with barrier method
    for (idx,mu) in enumerate(mus):

        eps = math.pow(10, -4)
        v0 = np.zeros(X.shape[0]) # lambd / X.shape[1] * np.reciprocal(np.max(np.abs(X), axis=1))
        seq_all = barr_method(Q, p, A, b, v0, eps, mu)

        opt_values = []
        for seq_cstep in seq_all: #seq_cstep : sequence of iterates for a centering step
            # computing only for the final iteration of the centering step (len(seq_cstep) : number of inner Newton iterations)
            opt_values += [objective(Q, p, seq_cstep[-1])] * len(seq_cstep)
        gaps[mu] = np.array(opt_values) - opt_values[-1]

    ax = plt.subplot()
    ax.set_yscale("log")
    for (idx,mu) in enumerate(mus):
        iters = np.arange(np.shape(gaps[mu])[0])
        plt.step(iters,gaps[mu],color=colors[idx],label="mu = "+str(mu))
    plt.ylabel("Gap f(v(t)) - f*")
    plt.xlabel("Newton iterations")
    plt.legend()
    plt.savefig("gaps.eps")
    plt.show()

main()
