#Convex Optimization -Homework 3
#Author : Yonatan Deloro

import numpy as np

def objective(Q,p,v):
    # return the value of the objective function of QP at v
    # (Q,p): parameters of the QP problem
    return v.T.dot(Q.dot(v)) + p.T.dot(v)

def objective_with_barrier(Q,p,A,b,t,v):
    # return the value of the objective function of QP + the log-barrier term at v
    # (Q,p,A,b): parameters of the QP problem ; t : log-barrier method parameter
    return t*(v.T.dot(Q.dot(v)) + p.T.dot(v)) - np.sum(np.log(b-A.dot(v)))

def backtracking_line_search(Q,p,A,b,t,v,grad_v,delta_v,alpha=0.25,beta=0.5):
    # returns the stepsize found with backtracing line search
    # at a given iteration of a centering step
    # (Q,p,A,b): parameters of the QP problem, t : log-barrier method parameter
    # v: estimated variable at an iteration
    # grad_v : gradient of the function at v
    # delta_v : search direction
    # alpha, beta : parameters of the backtracing line search technique (alpha \in (0,1/2), b \in (0,1))
    stepsize = 1
    oldf = objective_with_barrier(Q,p,A,b,t,v)
    #while (objective_with_barrier(Q, p, A, b, t, v + stepsize * delta_v) > oldf + alpha*stepsize*grad_v.T.dot(delta_v)) :
    while ((np.any(b - A.dot(v + stepsize * delta_v)<0))
           or (objective_with_barrier(Q, p, A, b, t, v + stepsize * delta_v) > oldf + alpha * stepsize * grad_v.T.dot(delta_v))):
        stepsize *= beta
    return stepsize

def centering_step(Q,p,A,b,t,v0,eps):
    # implements the Newton method to solve the centering step
    # (Q,p,A,b): parameter of the QP problem
    # t : log-barrier method parameter
    # v0 : initial v
    # eps  : target precision
    # outputs the sequence of variables iterates
    var_iters = [v0]
    while (True):
        v = var_iters[-1]
        tmp = np.reciprocal(b-A.dot(v))
        grad = t*(2*Q.dot(v)+p) + A.T.dot(tmp)
        Hess = t*2*Q + A.T.dot(np.diag(tmp*tmp)).dot(A)
        direction = -np.linalg.inv(Hess).dot(grad)
        if 0.5 * grad.T.dot(-direction) < eps:
            break
        stepsize = backtracking_line_search(Q,p,A,b,t,v,grad,+direction,alpha=0.25,beta=0.5)
        var_iters.append(v+stepsize*direction)
    return var_iters
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

def is_feasible(A,b,v):
    # returns True if v0 is feasible for the QP problem of parameters  (Q,p,A,b)
    return np.all(np.greater(b,A.dot(v)))

def barr_method(Q,p,A,b,v0,eps,mu):
    # implements the barrier method to solve QP problem of parameters (Q,p,A,b)
    # v0 : feasible point for initialization
    # eps  : target precision
    # mu : parameter of the barrier method
    # outputs the sequence of variables iterates as a sequence of sequences
    # (sequences of variable iterates for each centering step)

    m=A.shape[0] #number of contraints
    t=1.
    var_iters = [[v0]]
    while m/t > eps:
        v=var_iters[-1][-1]
        assert(is_feasible(A,b,v))
        var_iters_one_step = centering_step(Q,p,A,b,t,v,eps=eps)
        if len(var_iters_one_step)>1: #at least one Newton iteration in previous centering step
            var_iters.append(var_iters_one_step[1:])
        t *= mu
    return var_iters
