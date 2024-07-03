import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve

##################################### DF 1d #####################################

def DF_solve_full_array(n,h,F,A_eps,dA_eps):
    T = np.diag(-2*A_eps, k=0)
    T += np.diag(A_eps[0:n], k=1)
    T += np.diag(A_eps[1:n+1], k=-1)
    A1 = -T/h/h # type: ignore

    T = np.diag(dA_eps[0:n], k=1)
    T += -np.diag(dA_eps[1:n+1], k=-1)
    A2 = -T/2/h # type: ignore

    A = A1+A2
    A[0,1] = 0
    A[n,n-1] = 0
    A[0,0] = 1
    A[n,n] = 1
    b = np.concatenate(([0],F[1:n],[0]),axis=None)
    return np.linalg.solve(A,b)
def DF_solve_temp(n,h,F,A_eps,dA_eps):

    diag_A_eps = np.concatenate(([h*h],-2*A_eps[1:n],[h*h]),axis=None)
    diags_A_eps_p1 = np.concatenate(([0],A_eps[1:n]),axis=None)
    diags_A_eps_m1 = np.concatenate((A_eps[1:n],[0]),axis=None)

    A1 = diags_array(diag_A_eps, offsets=0)
    A1 += diags_array(diags_A_eps_p1, offsets=1)
    A1 += diags_array(diags_A_eps_m1, offsets=-1)
    A1 = -A1/h/h

    A2 = diags_array(dA_eps[0:n],offsets=1)
    A2 += diags_array(-dA_eps[1:n+1],offsets=-1)
    A2 = -A2/2/h


    A = A1+A2
    b = np.concatenate(([0],F[1:n],[0]),axis=None)
    return spsolve(A,b)
def DF_solve_laplacien(a_star,n,h,F):

    diag_A_eps = np.concatenate(([h*h],-2*a_star*np.ones(n-1),[h*h]),axis=None)
    diags_A_eps_p1 = np.concatenate(([0],a_star*np.ones(n-1)),axis=None)
    diags_A_eps_m1 = np.concatenate((a_star*np.ones(n-1),[0]),axis=None)

    A = diags_array(diag_A_eps, offsets=0)
    A += diags_array(diags_A_eps_p1, offsets=1)
    A += diags_array(diags_A_eps_m1, offsets=-1)
    A = -A/h/h

    b = np.concatenate(([0],F[1:n],[0]),axis=None)
    return spsolve(A,b)
def DF_solve(n,h,F,A_eps):
 
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
