import numpy as np
from numba import njit

@njit
def forward(A, B, pi, O):
    'Do the forward algorithm for the Baum-Welch.'
    T = len(O)             # Number of sequences
    N = A.shape[0]         # Number of states
    
    alpha_2dots = np.zeros((N, T)).astype(np.longdouble)
    alpha_hat = np.zeros_like(alpha_2dots).astype(np.longdouble)
    c = np.zeros(T).astype(np.longdouble)
    
    alpha_2dots[:,0] =  pi[:,0]*B[:, O[0]]
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
    'Do the backward algorithm for the Baum-Welch.'
    T = len(O)             # Number of sequences
    N = A.shape[0]         # Number of states
    
    beta_hat = np.zeros((N, T))
    beta_2dots = np.zeros_like(beta_hat)
    beta_2dots[:, -1] = 1
    beta_hat[:, -1] = c[-1]
    
    for t in range(T-2, -1, -1):
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += A[i, j] * B[j, O[t+1]] * beta_hat[j, t+1]
            beta_2dots[i, t] = temp
        beta_hat[:, t] = beta_2dots[:, t] * c[t]
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
def calculate_chi(K, N, T, observation_l, alpha_hat_l, beta_hat_l, c_l, chi_n, chi_d, l):
    'Calculates the necessarcy chi parameters which are needed for the update of B matrix'
    for k in range(K):
        for j in range(N):
            numerator = 0
            denominator = 0
            for t in range(T):
                if observation_l[t] == k:
                    numerator += (alpha_hat_l[j, t] * beta_hat_l[j, t]) / c_l[t]
                denominator += (alpha_hat_l[j, t] * beta_hat_l[j, t]) / c_l[t] 
            chi_n[l, j, k] = numerator
            chi_d[l, j, k] = denominator
    return


@njit
def bw(A, B, pi, O):
    """
    Does the Baum-welch iteration by using the helper functions and the forward-backward algorithms. 
    See http://www.cs.cmu.edu/~roni/11661/2017_fall_assignments/shen_tutorial.pdf for the underling 
    math for this function. The forward-backward algorithm is also constructed according to this
    papper written by Dawei Shen.
    """
    N = A.shape[0]         # Num of states
    K = B.shape[1]         # Num of observations
    num_sequence = O.shape[0]  # Num of sequences
    likelihood_of_sequence_holder = np.zeros(num_sequence)

    xi = np.zeros((num_sequence, N, N))

    chi_n = (np.zeros((num_sequence, N, K)))
    chi_d = np.zeros_like(chi_n)
    
    gamma = np.zeros((num_sequence, N))

    for l in range(num_sequence):
        observation_l = O[l]
        T = len(observation_l)
        likelihood_of_sequence, alpha_hat_l, c_l = forward(A, B, pi, observation_l)   # Call the forward algorithm to get the alpha and c:s
        likelihood_of_sequence_holder[l] = likelihood_of_sequence
        beta_hat_l = backward(A, B, pi, observation_l, c_l)      # Call the backward algorithm to get the beta:s
        calculate_xi_and_gamma(N, T, alpha_hat_l, beta_hat_l, c_l, A, B, observation_l, xi, gamma, l)   # Update the xi and gamma parameters
        calculate_chi(K, N, T, observation_l, alpha_hat_l, beta_hat_l, c_l, chi_n, chi_d, l)            # Update the chi parameter
 
    xi_ij = np.sum(xi, axis = 0)
    A = (np.divide(xi_ij.T, np.sum(xi_ij, axis=1))).T
    
    chi_n_sum = np.sum(chi_n, axis = 0)
    chi_d_sum = np.sum(chi_d, axis = 0)
    B = np.divide(chi_n_sum, chi_d_sum)

    gamma_j = np.sum(gamma, axis = 0)
    pi = gamma_j/(np.sum(gamma_j))
    pi = pi.reshape((len(pi), 1))

    likelihood_of_sequence = np.sum(likelihood_of_sequence_holder)/num_sequence    # The mean value of the likelihood of the observed sequence
    return A, B, pi, likelihood_of_sequence


def fit(A, B, O, K, TOL = 5e-3, NUM_ITER = 100, pi = 0):
    """
    Calls the baum-welch algorithm as many times as needed.
    A: the transition matrix of shape KxK given as a numpy array
    B: the emission matrix of shape KxL given as a numpy array
    O: the observations given of shape number_of_sequences x number of observations per sequence. Is a numpy array.
    K: number of states
    TOL: tolerance for stopping the iteration
    NUM_ITER: maximum number of iterations 
    pi: the initial transition matrix. If it is not given as input all the states are seen as equal probability.  

    To fit an HMM model simply call this function with the correct types and shapes explained above. 
    """
    delta = 10
    likelihood_list = []
    if type(pi) == int:  # if pi is not given start with equal probability on all the entries
        pi = np.array((np.ones(K)/K).reshape((-1, K))).T
    iter = 0
    while delta>TOL and iter<NUM_ITER:
        A, B, pi, prob_for_comp = bw(A, B, pi, O)
        likelihood_list.append(prob_for_comp)
        try:
            delta = likelihood_list[-1] - likelihood_list[-2]
        except:
            pass
        iter += 1
    return A, B, pi, likelihood_list, iter, delta
