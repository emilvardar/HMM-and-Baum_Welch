import numpy as np
from numba import njit

@njit
def forward(A, B, pi, O):
    T = len(O)
    N = A.shape[0]
    
    alpha_2dots = np.zeros((N, T))
    alpha_hat = np.zeros_like(alpha_2dots)
    c = np.zeros(T)
    
    alpha_2dots[:,0] =  pi[:,0]*(B[:, O[0]])
    c[0] = 1/(np.sum(alpha_2dots[:,0])) 
    alpha_hat[:,0] = c[0]*alpha_2dots[:,0]
    
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                alpha_2dots[i, t] +=  alpha_hat[j, t-1] * A[j, i] * B[i, O[t]]

        c[t] = 1/(np.sum(alpha_2dots[:,t]))
        alpha_hat[:, t] = alpha_2dots[:,t]*c[t]
    
    ln_P = np.sum(-np.log(c))
    return ln_P, alpha_hat, c


@njit
def backward(A, B, pi, O, c):
    T = len(O)
    N = A.shape[0]
    
    beta_hat = np.zeros((N, T))
    beta_hat[:, -1] = c[-1]
    
    for t in range(T-2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta_hat[i, t] +=  A[i, j] * B[j, O[t+1]] * beta_hat[j, t+1]
        beta_hat[:, t] = beta_hat[:, t] * c[t]
    return beta_hat

@njit
def bw(A, B, pi, O):
    N = A.shape[0]         # Num of states
    K = B.shape[1]         # Num of observations
    
    xi = np.zeros((O.shape[0], N, N))

    chi_n = (np.zeros((len(O), N, K)))
    chi_d = (np.zeros((len(O), N, K)))
    
    gamma = np.zeros((len(O), N))

    for l in range(O.shape[0]):
        T = len(O[l])
            
        _, alpha_hat_l, c_l = forward(A, B, pi, O[l])
        beta_hat_l = backward(A, B, pi, O[l], c_l)
        for i in range(N):
            for j in range(N):
                numerator = 0
                denominator = 0
                xi_temp = 0
                for t in range(T-1):
                    xi_temp += alpha_hat_l[i, t] * A[i,j] * B[j, O[l][t+1]] * beta_hat_l[j, t+1]
                xi[l, i, j] = xi_temp
        
        for j in range(N):
            gamma[l, j] = alpha_hat_l[j, 0] * beta_hat_l[j, 0] / c_l[0]
        
        
        for k in range(K):
            for j in range(N):
                numerator = 0
                denominator = 0
                for t in range(T):
                    if O[l][t] == k:
                        numerator += alpha_hat_l[j, t] * beta_hat_l[j, t] / c_l[t]
                    denominator += alpha_hat_l[j, t] * beta_hat_l[j, t] / c_l[t] 
                chi_n[l, j, k] = numerator
                chi_d[l, j, k] = denominator
 
    xi_ij = np.sum(xi, axis = 0)
    A = (np.divide(xi_ij.T, np.sum(xi_ij, axis=1))).T
    
    
    chi_n_sum = np.sum(chi_n, axis = 0)
    chi_d_sum = np.sum(chi_d, axis = 0)
    B = np.divide(chi_n_sum, chi_d_sum)

    gamma_j = np.sum(gamma, axis = 0)
    pi = gamma_j/(np.sum(gamma_j))
    pi = pi.reshape((len(pi), 1))
    
    return A, B, pi

def fit(A, B, O, K, TOL = 1e-6, NUM_ITER = 5000, pi = 0):
    diff_A = 100
    diff_B = 100
    A_old = 0
    B_old = 0
    if type(pi) == int:
        pi = np.array((np.ones(K)/K).reshape((-1, K))).T
    iter = 0
    while (diff_A>TOL or diff_B>TOL) and iter<NUM_ITER:
        A, B, pi = bw(A, B, pi, O)
        diff_A = np.linalg.norm(A - A_old)/(K**2)
        diff_B = np.linalg.norm(B - B_old)/(K**2)
        A_old = A
        B_old = B
        iter += 1
    return A, B, pi