import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve

##################################### Finite difference 1d #####################################

def DF_solve_laplacien(a,n,h,F):

    diag_A_eps = np.concatenate(([h*h],-2*np.ones(n-1),[h*h]),axis=None)
    diags_A_eps_p1 = np.concatenate(([0],np.ones(n-1)),axis=None)
    diags_A_eps_m1 = np.concatenate((np.ones(n-1),[0]),axis=None)

    A = diags_array(diag_A_eps, offsets=0)
    A += diags_array(diags_A_eps_p1, offsets=1)
    A += diags_array(diags_A_eps_m1, offsets=-1)
    A = -A

    b = h*h/a*np.concatenate(([0],F[1:n],[0]),axis=None)
    return spsolve(A,b)

def DF_solve_bis(n,h,F,A_eps):
 
    diag_A_eps = np.concatenate(([h*h],-2*A_eps[1:n],[h*h]))
    diags_A_eps_p1 = np.concatenate(([0],A_eps[1:n]))
    diags_A_eps_m1 = np.concatenate((A_eps[1:n],[0]))

    A1 = diags_array(diag_A_eps, offsets=0)
    A1 += diags_array(diags_A_eps_p1, offsets=1)
    A1 += diags_array(diags_A_eps_m1, offsets=-1)
    A1 = -A1/h/h

    diags_dA_eps_p1 = np.concatenate(([0],(A_eps[2:n+1]-A_eps[0:n-1])/2/h))
    diags_dA_eps_m1 = np.concatenate(((A_eps[2:n+1]-A_eps[0:n-1])/2/h,[0]))
    A2 = diags_array(diags_dA_eps_p1,offsets=1)
    A2 += diags_array(-diags_dA_eps_m1,offsets=-1)
    A2 = -A2/2/h


    A = A1+A2
    b = np.concatenate(([0],F[1:n],[0]))
    return spsolve(A,b)

def DF_solve(n,h,F,A_eps):
 
    diag_A_eps = np.concatenate(([4*h*h],[-A_eps[2]-4*A_eps[0]],(-A_eps[3:n]-A_eps[1:n-2]),[-A_eps[n-2]-4*A_eps[n]],[4*h*h]))
    diags_A_eps_p2 = np.concatenate(([0],[A_eps[2]],A_eps[3:n]))
    diags_A_eps_m2 = np.concatenate((A_eps[1:n-2],[A_eps[n-2]],[0]))
    diags_A_eps_p1 = np.concatenate(([0],[A_eps[0]],np.zeros(n-3),[3*A_eps[n]]))
    diags_A_eps_m1 = np.concatenate(([3*A_eps[0]],np.zeros(n-3),[A_eps[n]],[0]))


    A = diags_array(diag_A_eps, offsets=0)
    A += diags_array(diags_A_eps_p2, offsets=2)
    A += diags_array(diags_A_eps_m2, offsets=-2)
    A += diags_array(diags_A_eps_p1, offsets=1)
    A += diags_array(diags_A_eps_m1, offsets=-1)

    A = -A/4/h/h


    b = np.concatenate(([0],F[1:n],[0]))
    return spsolve(A,b)