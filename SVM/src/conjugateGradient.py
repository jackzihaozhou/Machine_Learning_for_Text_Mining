import sys
import numpy as np
import math
import scipy.sparse as sparse
from scipy.sparse import csr_matrix,hstack

# for realsim dataset
# lambda_ = 7230.875

# for covtype dataset
# lambda_ = 3631.3203125


# please refer wikipedea
# https://en.wikipedia.org/wiki/Conjugate_gradient_method
# this function solve  Hd = -g



def conjugateGradient(X, I, grad, lambda_, tol=1e-1, max_iter=100):
    """conjugateGradient

    :param X: shape = (N, M)
    :param I: can be a binary vector, shape = (N,),
              or a list of indices as defined in the handout.
    :param grad: shape = (M,)
    :param lambda_:
    :param tol:
    :param max_iter:
    """
    # Hessian vector product
    def Hv(X, I, v):
        ret = v + (2.0 * lambda_ / X.shape[0]) * X[I].transpose().dot(X[I].dot(v))
        return ret.reshape(-1,1)

    # initial
    #print("in conjugateGradient")
    d = np.random.rand(X.shape[1]).reshape(-1,1)
    #print("1111")
    d = csr_matrix(d)
    print(f"d.shape :{d.shape}")
    Hd = Hv(X, I, d)
    print(f"Hd.shape :{Hd.shape}")
    print(f"grad.shape :{grad.shape}")
    r = -grad - Hd
    p = r
    #print(f"r.shape :{r.shape}")
    rsold = r.T.dot(r)
    #print("3333")
    for cg_iter in range(max_iter):
        print(f"cg_iter : {cg_iter}")
        Ap = Hv(X, I, p)
        alpha = rsold / p.T.dot(Ap)
        #print(f"alpha.shape :{alpha.shape}")
        d = d + alpha * p
        #print(f"d.shape :{d.shape}")
        r = r - alpha * Ap
        #print(f"r.shape :{r.shape}")
        rsnew = r.T.dot(r)
        #print(f"rsnew.shape :{rsnew.shape}")
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        #print(f"p.shape :{p.shape}")
        rsold = rsnew

    return d, cg_iter + 1
