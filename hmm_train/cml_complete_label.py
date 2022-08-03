"""
This code is written by Emil Vardar (student at KTH Royal Institute of Technology,
Stockholm, Sweden). And the goal is to update transition and emission matrices for 
Hidden Markov Models (HMM) using conditional maximum likelihood approach (CML) where 
the labels for the states are complete. 

It is open source. Use at your own risk.

The reference [1] refered many times below is:
[1]: Riis, Søren. Hidden Markov models and neural 
     networks for speech recognition. Technical University of 
     Denmark [Department of Mathematical Modeling], 1998.
"""

import numpy as np
from numba import njit
from sequence_creator_according_to_hmm_model import *
import matplotlib.pyplot as plt 

@njit
def forward(A, B, pi, O):
    'Do the forward algorithm for the Baum-Welch.'
    T = len(O)             # Number of sequences
    N = A.shape[0]         # Number of states

    alpha_2dots = np.zeros((N, T))
    alpha_hat = np.zeros_like(alpha_2dots)
    c = np.zeros(T)
    
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
        alpha_hat[:, t] = alpha_2dots[:,t] * c[t]

    ln_P = np.sum(-np.log(c))
    return ln_P, alpha_hat, c

@njit
def backward(A, B, O, c):
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
def calculate_n_ijl(A, B, alpha, beta, O):
    """Calculate n_(ij)(l) according to equation 2.46 in [1]."""
    N = A.shape[0]
    L = alpha.shape[1]
    n_ijl = np.zeros((N, N, L-1))
    for i in range(N):
        for j in range(N):
            for l in range(L-1):
                n_ijl[i, j, l] = alpha[i, l] * A[i, j] * B[j, O[l+1]] * beta[j, l+1]
    return n_ijl

@njit
def calculate_n_il(alpha, beta, c):
    """Calculate n_i(l) according to equation 2.47 in [1]."""
    N = alpha.shape[0]
    L = alpha.shape[1]
    n_il = np.zeros((N, L))
    for i in range(N):
        for l in range(L):
            n_il[i, l] = alpha[i, l] * beta[i, l] / c[l]
    return n_il

@njit
def modified_forward(A, B, pi, observations, states):
    """
    Do the modified forward algorithm according to [1]. (Page 53).
    Observe that here I ignore the termination step since the impact of it
    is minimal.
    """
    L = len(observations)        # Sequence length
    N = A.shape[0]               # Number of states
    
    alpha_dots = np.zeros((N, L))
    alpha_tilde = np.zeros_like(alpha_dots)
    c = np.zeros(L)
    
    # Initialization step in Algorithm 4.1 given in [1].
    for j in range(N):
        if states[0] == j:
            alpha_dots[j, 0] = B[j, observations[0]] * pi[j, 0]

    c[0] = 1/(np.sum(alpha_dots[:,0])) 
    alpha_tilde[:,0] = c[0]*alpha_dots[:,0]
    
    # Recursion step in Algorithm 4.1 given in [1].
    for l in range(1, L): 
        for j in range(N):
            if states[l] == j:
                temp = 0
                for i in range(N):
                    temp += alpha_tilde[i, l-1] * A[i, j]
                alpha_dots[j, l] = temp * B[j, observations[l]]
        c[l] = 1/(np.sum(alpha_dots[:,l]))
        alpha_tilde[:, l] = alpha_dots[:,l] * c[l]

    ln_P = np.sum(-np.log(c))
    return ln_P, alpha_tilde, c

@njit
def modified_backward(A, B, observations, states, c):
    """
    Do the modified backward algorithm according to [1]. Similar to the 
    modified_forward function I again ignore the termination step. 
    """
    L = len(observations)  # Sequence length
    N = A.shape[0]         # Number of states
    
    beta_tilde = np.zeros((N, L))
    beta_dots = np.zeros_like(beta_tilde)

    # Initialization step according to more efficient implementation of
    # Algorithm 4.2 given in [1]. This implementation is given in the text 
    # below algorithm 4.2. 
    for i in range(N):
        if i == states[-1]:
            beta_dots[i, L-1] = 1
    beta_tilde[:, L-1] = beta_dots[:, L-1] * c[L-1] 

    # Recursion step according to more efficient implementation of
    # Algorithm 4.2 given in [1].
    for l in range(L-2, -1, -1):
        for i in range(N):
            if states[l] == i:
                temp = 0
                for j in range(N):
                    temp += beta_tilde[j, l+1] * A[i, j] * B[j, observations[l+1]]
                beta_dots[i, l] = temp
        beta_tilde[:, l] = beta_dots[:, l] * c[l]
    return beta_tilde

@njit
def calculate_m_ijl(A, B, alpha_tilde, beta_tilde, observations, states):
    """Calculate m_(ij)(l) according to equation 4.7 in [1]."""
    N = A.shape[0]              # Number of states
    L = alpha_tilde.shape[1]    # Length of each sequence
    m_ijl = np.zeros((N, N, L-1))
    for i in range(N): 
        for j in range(N):
            for l in range(L-1):
                if states[l+1] == j: 
                    m_ijl[i, j, l] = alpha_tilde[i, l] * A[i, j] * B[j, observations[l+1]] * beta_tilde[j, l+1]
    return m_ijl

@njit
def calculate_m_il(alpha_tilde, beta_tilde, c_tilde):
    """Calculate m_i(l) according to equation 4.8 in [1]."""
    return calculate_n_il(alpha_tilde, beta_tilde, c_tilde)

@njit
def sum_axis_2(x):
    """Sum a 3 dimensional matrix in axis=2 direction."""
    return np.sum(x, axis=2)

@njit
def calculate_x_ia(x_il, observations, B):
    """Calculate x_i(a) according to equation 4.22 given in [1].
    The input should either be m_i(l) or n_i(l). Hence here x can
    take two values x={m,n}"""
    x_ia = np.zeros_like(B)
    for i in range(B.shape[0]):
        for a in range(B.shape[1]):
            temp = 0
            for l in range(x_il.shape[1]):
                if observations[l] == a:
                    temp += x_il[i, l]
            x_ia[i, a] = temp
    return x_ia

@njit
def forward_backward(A, B, pi, observations, states):
    """Calculate the alpha, beta and c's with the basic forward-backward functions
    and the modified forward-backward functions according to [1]."""
    _, alpha_tilde, c_tilde = modified_forward(A, B, pi, observations, states)
    beta_tilde = modified_backward(A, B, observations, states, c_tilde)
    prob, alpha, c = forward(A, B, pi, observations)
    beta = backward(A, B, observations, c)
    return alpha_tilde, beta_tilde, alpha, beta, prob, c_tilde, c

@njit 
def dz_trans(A, m_ij, n_ij):
    """Calculate the derivative of negative log conditional likelihood w.r.t. the 
    auxiliary variables z_ij (which are auxiliary for updating the transition 
    matrix A) according to 4.27 in [1]."""
    dz = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            dz[i,j] = -(m_ij[i,j] - n_ij[i,j] - A[i,j] * (np.sum(m_ij[i,:]) - np.sum(n_ij[i,:])))
    return dz

@njit 
def dz_emis(B, m_ia, n_ia):
    """Calculate the derivative of negative log conditional likelihood w.r.t. the 
    auxiliary variables z_ij (which are auxiliary for updating the emission matrix 
    B)."""
    dz = np.zeros_like(B)
    for i in range(B.shape[0]):
        for a in range(B.shape[1]):
            dz[i,a] = -(m_ia[i, a] + n_ia[i,a] - B[i,a] * (np.sum(m_ia[i,:]) + np.sum(n_ia[i,:])))
    return dz

@njit
def calculate_derivatives(A, B, pi, observations, states):
    """A function that gathers all the important variables and returns the derivatives of the 
    negative log conditional likelihood w.r.t. the auxiliary variables for updating transition
    and emission matrices."""
    alpha_tilde, beta_tilde, alpha, beta, prob, c_tilde, c = forward_backward(A, B, pi, observations, states)
    m_ijl = calculate_m_ijl(A, B, alpha_tilde, beta_tilde, observations, states)
    m_ij = sum_axis_2(m_ijl)
    m_il = calculate_m_il(alpha_tilde, beta_tilde, c_tilde)
    m_ia = calculate_x_ia(m_il, observations, B)

    n_ijl = calculate_n_ijl(A, B, alpha, beta, observations)
    n_ij = sum_axis_2(n_ijl)
    n_il = calculate_n_il(alpha, beta, c)
    n_ia = calculate_x_ia(n_il, observations, B)

    dz_t = dz_trans(A, m_ij, n_ij)
    dz_e = dz_emis(B, m_ia, n_ia)

    return dz_t, dz_e, prob

@njit
def update_trans(delta_z, A, eta):
    """Updates the transition matrix according to 4.26 in [1]. Observe that
    we also have eta here for step length and a negative that has been forgotten 
    in [1]."""
    numerator = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            numerator[i, j] = A[i, j] * np.exp(-eta*delta_z[i, j])
    denominator = np.sum(numerator, axis=1).reshape(A.shape[0], 1)
    A_new = numerator/denominator
    return A_new

@njit
def update_emis(delta_z, B, eta):
    """Updates the emission matrix."""
    numerator = np.zeros_like(B)
    for i in range(B.shape[0]):
        for a in range(B.shape[1]):
            numerator[i, a] = B[i, a] * np.exp(-eta*delta_z[i, a])
    denominator = np.sum(numerator, axis=1).reshape(B.shape[0], 1)
    B_new = numerator/denominator
    return B_new

@njit
def update_1_epoch(A, B, pi_gt, teoretical_observations, teoretical_states, ETA):
    """Updates the transition and emission matrices for 1 epoch of training data"""
    prob_temp = np.zeros(len(teoretical_observations))
    for i in range(len(teoretical_observations)):  # Go through all the training sequences and do online update
        dz_t, dz_e, prob = calculate_derivatives(A, B, pi_gt, teoretical_observations[i], teoretical_states[i])
        A = update_trans(dz_t, A, ETA)
        B = update_emis(dz_e, B, ETA)
        prob_temp[i] = prob
    return np.sum(prob_temp)/len(teoretical_observations), A, B

@njit
def fit(A, B, pi_gt, teoretical_observations, teoretical_states, ETA, epochs):
    """Updates the transition and emission matrices for 'epoch' number of times."""
    prob_list = []
    for epoch in range(epochs):
        prob_efter_epoch, A, B = update_1_epoch(A, B, pi_gt, teoretical_observations, teoretical_states, ETA)
        prob_list.append(prob_efter_epoch)
    return A, B, prob_list
