import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve


##################################### Computation about a* 1d #####################################

def compute_a_star(A_eps):
    M = 1/A_eps
    return 1/np.mean(M)
def mean_a(A_eps):
    return np.mean(A_eps)
def compute_a_derive(A_eps,h):
    n = A_eps.size-1
    return np.concatenate(([(-A_eps[2]+4*A_eps[1]-3*A_eps[0])/2/h],(A_eps[2:n+1]-A_eps[0:n-1])/2/h,[(3*A_eps[n]-4*A_eps[n-1]+A_eps[n-2])/2/h]))

def compute_corrected_a(n,h,A_eps):
    w = DF_solve_corrector(n,h,A_eps)
    dw = compute_a_derive(w,h)
    return np.mean(A_eps*(dw+1))


def DF_solve_corrector(n,h,A_eps):

    diag_A_eps = np.concatenate(([h*h],-2*A_eps[1:n],[h*h]),axis=None)
    diags_A_eps_p1 = np.concatenate(([0],A_eps[1:n]),axis=None)
    diags_A_eps_m1 = np.concatenate((A_eps[1:n],[0]),axis=None)
    A1 = diags_array(diag_A_eps, offsets=0)
    A1 += diags_array(diags_A_eps_p1, offsets=1)
    A1 += diags_array(diags_A_eps_m1, offsets=-1)
    A1 = -A1/h/h

    diags_dA_eps_p1 = np.concatenate(([0],(A_eps[2:n+1]-A_eps[0:n-1])/2/h),axis=None)
    diags_dA_eps_m1 = np.concatenate(((A_eps[2:n+1]-A_eps[0:n-1])/2/h,[0]),axis=None)
    A2 = diags_array(diags_dA_eps_p1,offsets=1)
    A2 += diags_array(-diags_dA_eps_m1,offsets=-1)
    A2 = -A2/2/h

    A = A1+A2
    b = np.concatenate(([0],(A_eps[2:n+1]-A_eps[0:n-1])/2/h,[0]),axis=None)
    return spsolve(A,b)