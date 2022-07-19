"""This one uses the 'correct' update rule for B."""

import numpy as np
from numba import njit

@njit
def forward(A, B, pi, O):
    'Does the forward algorithm for Baum-Welch.'
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
            temp = 0
            for j in range(N):
                temp += alpha_hat[j, t-1] * A[j, i]
            alpha_2dots[i, t] = temp * B[i, O[t]]        
        c[t] = 1/(np.sum(alpha_2dots[:,t]))
        alpha_hat[:, t] = alpha_2dots[:,t]*c[t]
    
    ln_P = np.sum(-np.log(c))
    return ln_P, alpha_hat, c


@njit
def backward(A, B, pi, O, c):
    'Does the backward algorithm for Baum-Welch.'
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
def calculate_xi_and_gamma(N, T, alpha_hat_l, beta_hat_l, c_l, A, B, observation_l, xi, gamma, l):
    'Calculates the necessarcy xi and gamma parameters which are needed for the update of A and pi matrices'
    for i in range(N):
        for j in range(N):
            xi_temp = 0
            for t in range(T-1):
                xi_temp += alpha_hat_l[i, t] * A[i,j] * B[j, observation_l[t+1]] * beta_hat_l[j, t+1]
            xi[l, i, j] = xi_temp
    
        gamma[l, i] = alpha_hat_l[i, 0] * beta_hat_l[i, 0] / c_l[0]
    return

@njit
def calculate_chi(K, N, D, T, observation_l, alpha_hat_l, beta_hat_l, c_l, chi_n, chi_d, l):
    'Calculates the necessarcy chi parameters which are needed for updating the B matrix'
    for k in range(K):       # Go through all the observation states (actually L)
        for j in range(0, N, D):   # Go thorugh all the static states (that is S=j and D=whatever)
            numerator = 0
            denominator = 0
            for t in range(T):
                for d in range(D):
                    if observation_l[t] == k:
                        numerator += (alpha_hat_l[j+d, t] * beta_hat_l[j+d, t]) / c_l[t]
                    denominator += (alpha_hat_l[j+d, t] * beta_hat_l[j+d, t]) / c_l[t] 
            chi_n[l, j//D, k] = numerator
            chi_d[l, j//D, k] = denominator
    return


@njit
def update_B(B, N, D, K):
    B_new = np.zeros((N, K))
    for i in range(0, N, D):
        row_i = B[i//D, :]
        for d in range(D):
            B_new[i+d, :] = row_i
    return B_new

@njit
def bw_edhmm(A, B, pi, O, D):
    N = A.shape[0]         # Num of states
    K = B.shape[1]         # Num of observation-states
    num_sequence = O.shape[0]  # Num of sequences
    likelihood_of_sequence_holder = np.zeros(num_sequence)

    xi = np.zeros((num_sequence, N, N))
    chi_n = np.zeros((num_sequence, N//D, K))
    chi_d = np.zeros((num_sequence, N//D, K))
    gamma = np.zeros((num_sequence, N))

    for l in range(num_sequence):
        observation_l = O[l]
        T = len(observation_l)
            
        likelihood_of_sequence, alpha_hat_l, c_l = forward(A, B, pi, observation_l)   # Call the forward algorithm to get the alpha and c:s
        likelihood_of_sequence_holder[l] = likelihood_of_sequence
        beta_hat_l = backward(A, B, pi, observation_l, c_l)

        calculate_xi_and_gamma(N, T, alpha_hat_l, beta_hat_l, c_l, A, B, observation_l, xi, gamma, l)   # Update the xi and gamma parameters
        calculate_chi(K, N, D, T, observation_l, alpha_hat_l, beta_hat_l, c_l, chi_n, chi_d, l)            # Update the chi parameter
 
    xi_ij = np.sum(xi, axis = 0)
    A = (np.divide(xi_ij.T, np.sum(xi_ij, axis=1))).T
    
    chi_n_sum = np.sum(chi_n, axis = 0)
    chi_d_sum = np.sum(chi_d, axis = 0)
    B = np.divide(chi_n_sum, chi_d_sum)
    B = update_B(B, N, D, K)
    
    gamma_j = np.sum(gamma, axis = 0)
    pi = gamma_j/(np.sum(gamma_j))
    pi = pi.reshape((len(pi), 1))

    likelihood_of_sequence = np.sum(likelihood_of_sequence_holder)/num_sequence    # The mean value of the likelihood of the observed sequence

    return A, B, pi, likelihood_of_sequence


def fit_edhmm(A, B, pi, O, D, K, L, TOL = 1e-6, NUM_ITER = 5000, printer=True, print_modulo=100):
    delta = 10
    likelihood_list = []
    iter = 0
    while delta>TOL and iter<NUM_ITER:
        A, B, pi, prob_for_comp = bw_edhmm(A, B, pi, O, D)
        likelihood_list.append(prob_for_comp)
        try:
            delta = likelihood_list[-1] - likelihood_list[-2]
        except:
            pass
        iter += 1
    return A, B, pi, likelihood_list, iter, delta





