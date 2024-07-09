import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve


##################################### Computation about a* 1d #####################################

def compute_a_star(A_per):
    return 1/(np.mean(1/A_per))

def compute_a_derive(A,h):
    n = A.size-1
    return np.concatenate(([(-A[2]+4*A[1]-3*A[0])/2/h],(A[2:n+1]-A[0:n-1])/2/h,[(3*A[n]-4*A[n-1]+A[n-2])/2/h]))

def compute_corrected_a(n,h,A_eps):
    w = DF_solve_corrector(n,h,A_eps)
    dw = compute_a_derive(w,h)
    return np.mean(A_eps*(dw+1))

def DF_solve_corrector(n,h,A_per):

    diag_A_eps = np.concatenate(([4*h*h],[-A_per[2]-4*A_per[0]],(-A_per[3:n]-A_per[1:n-2]),[-A_per[n-2]-4*A_per[n]],[4*h*h]))
    diags_A_eps_p2 = np.concatenate(([0],[A_per[2]],A_per[3:n]))
    diags_A_eps_m2 = np.concatenate((A_per[1:n-2],[A_per[n-2]],[0]))
    diags_A_eps_p1 = np.concatenate(([0],[A_per[0]],np.zeros(n-3),[3*A_per[n]]))
    diags_A_eps_m1 = np.concatenate(([3*A_per[0]],np.zeros(n-3),[A_per[n]],[0]))

    A = diags_array(diag_A_eps, offsets=0)
    A += diags_array(diags_A_eps_p2, offsets=2)
    A += diags_array(diags_A_eps_m2, offsets=-2)
    A += diags_array(diags_A_eps_p1, offsets=1)
    A += diags_array(diags_A_eps_m1, offsets=-1)
    A += diags_array([-4*h*h], offsets=-n)
    A = -A/4/h/h

    b = np.concatenate(([0],(A_per[2:n+1]-A_per[0:n-1])/2/h,[0]),axis=None)
    return spsolve(A,b)