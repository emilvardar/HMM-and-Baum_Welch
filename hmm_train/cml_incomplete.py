"""
This code is written by Emil Vardar (student at KTH Royal Institute of Technology,
Stockholm, Sweden). And the goal is to update transition, emission and label distribution 
matrices for Hidden Markov Models (HMM) using conditional maximum likelihood approach (CML) 
where the labels for the states are 'incomplete'. 

It is open source. Use at your own risk.

The reference [1] refered many times below is:
[1]: Riis, Søren. Hidden Markov models and neural 
     networks for speech recognition. Technical University of 
     Denmark [Department of Mathematical Modeling], 1998.
"""

import numpy as np
from numba import njit

@njit
def create_zeros(N, L, S):
    """Defines zero-matrices to be used in the forward-backward algorithm."""
    alpha = np.zeros((N, L, S+1))
    alpha_p = np.zeros_like(alpha)
    alpha_m = np.zeros_like(alpha)

    alpha_norm = np.zeros_like(alpha)  
    alpha_p_norm = np.zeros_like(alpha)
    alpha_m_norm = np.zeros_like(alpha)

    c = np.zeros((L, S+1))
    c_p = np.zeros_like(c)
    c_m = np.zeros_like(c)
    return alpha, alpha_p, alpha_m, alpha_norm, alpha_p_norm, alpha_m_norm, c, c_p, c_m

@njit
def normalizer(c, alpha, alpha_norm, l, s):
    """Normalize alpha in axis=0 direction and save the 1 over the summation in the c matrix."""
    if np.sum(alpha[:,l,s]) != 0:
        c[l,s] = 1/np.sum(alpha[:,l,s])
        alpha_norm[:,l,s] = alpha[:,l,s] * c[l,s]
    return c, alpha_norm

@njit
def forward_initialization(B, P, pi, N, observations, labels, alpha, alpha_m, alpha_p, alpha_norm, alpha_m_norm, alpha_p_norm, c, c_m, c_p):
    """This function does the normalized version of the initialization step in Algorithm B.3 given in [1]."""
    for j in range(N):
        joint = B[j, observations[0]] * pi[j, 0]
        alpha_m[j, 0, 0] = joint * P[j, N]
        alpha_p[j, 0, 1] = joint * P[j, labels[0]] 
    alpha = alpha_p + alpha_m

    c, alpha_norm = normalizer(c, alpha, alpha_norm, 0, 0)
    c_m, alpha_m_norm = normalizer(c_m, alpha_m, alpha_m_norm, 0, 0)
    c_p, alpha_p_norm = normalizer(c_p, alpha_p, alpha_p_norm, 0, 1)
    c, alpha_norm = normalizer(c, alpha, alpha_norm, 0, 1)
    return c, c_m, c_p, alpha_norm, alpha_m_norm, alpha_p_norm

@njit
def forward_recursion(L, S, N, A, B, P, observations, labels, alpha, alpha_p, alpha_m, c, c_p, c_m, alpha_norm, alpha_p_norm, alpha_m_norm):
    """This function does the normalized version of the recursion step in given in Algorithm B.3 given in [1]."""
    for l in range(1, L):
        for s in range(S+1):
            for j in range(N):
                temp_p = 0
                temp_m = 0
                for i in range(N):
                    a_ij = A[i, j]
                    if s != 0:
                        temp_p += a_ij * alpha_norm[i, l-1, s-1]
                    temp_m += a_ij * alpha_norm[i, l-1, s]
                b_jl = B[j, observations[l]]
                alpha_p[j, l, s] = b_jl * P[j, labels[s-1]] * temp_p   
                alpha_m[j, l, s] = b_jl * P[j, N] * temp_m 
                alpha[j, l, s] = alpha_p[j, l, s] + alpha_m[j, l, s]
            c, alpha_norm = normalizer(c, alpha, alpha_norm, l, s)
            c_p, alpha_p_norm = normalizer(c_p, alpha_p, alpha_p_norm, l, s)
            c_m, alpha_m_norm = normalizer(c_m, alpha_m, alpha_m_norm, l, s)
    return alpha_norm, alpha_p_norm, alpha_m_norm, c, c_p, c_m

@njit
def incomplete_label_forward(A, B, P, pi, observations, labels):
    """
    This function implements the forward algorithm given in Algorithm B.3 
    in [1] (however this is a normalized version of that, for numerical 
    reasons). This forward algorithm is for the special case where we 
    have incomplete labels (S<L, where S is the number of labels and L 
    is the number of observaiton values). 
    """
    L = len(observations)    # Number of observations
    S = len(labels)          # Number of labels (S<L) 
    N = A.shape[0]           # Number of states 

    # Create zero-matrices to be used later
    alpha, alpha_p, alpha_m, alpha_norm, alpha_p_norm, alpha_m_norm, c, c_p, c_m = create_zeros(N, L, S)
    
    # Initialization step of Alg B.3 in [1]
    c, c_m, c_p, alpha_norm, alpha_m_norm, alpha_p_norm = forward_initialization(
        B, P, pi, N, observations, labels, alpha, alpha_m, alpha_p, alpha_norm, alpha_m_norm, alpha_p_norm, c, c_m, c_p)

    # Recursion step of Alg B.3 in [1]
    alpha_norm, alpha_p_norm, alpha_m_norm, c, c_p, c_m = forward_recursion(
        L, S, N, A, B, P, observations, labels, alpha, alpha_p, alpha_m, c, c_p, c_m, alpha_norm, alpha_p_norm, alpha_m_norm)
    return alpha_norm, alpha_p_norm, alpha_m_norm, c, c_p, c_m

@njit
def backward_recursion(L, S, N, beta_norm, beta, c, A, B, P,labels, observations):
    """This function does the normalized version of the recursion step in given in Algorithm B.4 given in [1]."""
    for l in range(L-2, -1, -1):
        for s in range(S, -1, -1):
            for i in range(N):
                temp_p = 0
                temp_m = 0
                for j in range(N):
                    if s != S:
                        temp_p += beta_norm[j, l+1, s+1] * A[i, j] * P[j, labels[s]] * B[j, observations[l+1]] 
                    temp_m += beta_norm[j, l+1, s] * A[i, j] * P[j, N] * B[j, observations[l+1]]
                beta[i, l, s] = temp_p + temp_m
            beta_norm[:, l, s] = beta[:, l, s] * c[l, s]
    return beta_norm

@njit
def incomplete_label_backward(A, B, P, observations, labels, c):
    """
    This function implements the backward algorithm given in Algorithm B.4 in [1] but
    with a slight normalization change to prevent numerical issues. 
    This backward algorithm is for the special case where we have incomplete labels (S<L, where S is the
    number of labels and L is the number of observaiton values).
    """
    L = len(observations)    # Number of observations
    S = len(labels)          # Number of labels S<L. 
    N = A.shape[0]           # Number of states

    beta = np.zeros((N, L, S+1))
    beta_norm = np.zeros_like(beta)

    # Initializaiton step
    beta[:, L-1, S] = 1
    beta_norm[:, L-1, S] = beta[:, L-1, S] * c[L-1, S]
    
    # Recursion step
    beta_norm = backward_recursion(L, S, N, beta_norm, beta, c, A, B, P,labels, observations)
    return beta_norm

@njit
def calculate_m_il(alpha, alpha_p, alpha_m, beta, c_p, c_m):
    """
    Calculates the m_i^+(l, s) and m_i^-(l, s) according to equation B.21 and B.22
    in [1]. Furthermore, calculates the m_i(l) by summing these two according to
    equation B.23.
    """    
    N = alpha.shape[0]    # Number of states
    L = alpha.shape[1]    # Number of observations
    S = alpha.shape[2]-1  # Number of labels S<L. 
    
    m_ils_p = np.zeros_like(alpha)    # The positive part of m_i(l, s) --> m_i+(l,s)
    m_ils_m = np.zeros_like(alpha)    # The negative part of m_i(l, s) --> m_i-(l,s)
    for i in range(N):
        for l in range(L):
            for s in range(S+1):
                if c_p[l,s] !=0:
                    m_ils_p[i, l, s] = alpha_p[i, l, s] * beta[i, l, s] / c_p[l,s] 
                if c_m[l,s] != 0:
                    m_ils_m[i, l, s] = alpha_m[i, l, s] * beta[i, l, s] / c_m[l,s]   
    # Sum m_i^+(l, s) and m_i^-(l, s) 
    m_il = np.sum((m_ils_p + m_ils_m), axis=2)
    return m_il, m_ils_p, m_ils_m

@njit
def calculate_m_ijl(A, B, P, observations, labels, alpha, beta):
    """Calculates m_ij(l) according to equation B.24 in [1]."""    
    N = alpha.shape[0]    # Number of states
    L = alpha.shape[1]    # Number of observations
    S = alpha.shape[2]-1  # Number of labels S<L. 
    m_ijl = np.zeros((N, N, L))

    # Then calculate the numerator and divide with correct denominator calcualted above
    for j in range(N):
        for i in range(N):
            for l in range(1, L):
                temp_p = 0
                temp_m = 0
                for s in range(S+1):
                    if s != 0:
                        temp_p += alpha[i, l-1, s-1] * B[j, observations[l]] * P[j, labels[s-1]] * A[i, j] * beta[j, l, s]
                    temp_m += alpha[i, l-1, s] * B[j, observations[l]] * P[j, N] * A[i, j] * beta[j, l, s]
                m_ijl[i, j, l] = temp_p + temp_m
    return m_ijl

@njit
def calculate_m_ic(m_ils_p, m_ils_m, labels):
    """Calculates m_i(c) according to equation B.26 in [1]. m_i(c) is used to
    update the label matrix P."""
    N = m_ils_p.shape[0]    # Number of states
    L = m_ils_p.shape[1]    # Number of observations
    S = m_ils_p.shape[2]-1  # Number of labels S<L. 
    
    m_ic = np.zeros((N, N+1))
    for i in range(N):
        for c in range(N+1):
            temp = 0
            if c < N:
                for l in range(L):
                    for s in range(S+1):
                        if s> 0 and c == labels[s-1]:
                            temp += m_ils_p[i, l, s]
                m_ic[i, c] = temp
            else:
                for l in range(L):
                    for s in range(S+1):
                        temp += m_ils_m[i, l, s]
                m_ic[i, c] = temp
    return m_ic

@njit
def calculate_m_ia(m_il, B, observations):
    m_ia = np.zeros_like(B)
    for i in range(B.shape[0]):
        for a in range(B.shape[1]):
            temp = 0
            for l in range(m_il.shape[1]):
                if observations[l] == a:
                    temp += m_il[i, l]
            m_ia[i, a] = temp 
    return m_ia


def update_matrices(A, B, P, pi, observations, labels, update_A, update_B, update_P):
    m_ij = np.zeros_like(A)
    m_ia = np.zeros_like(B)
    m_ic = np.zeros_like(P)
    for r in range(len(observations)):
        # Calculate the necessary alpha and beta matrices
        alpha, alpha_p, alpha_m, c, c_p, c_m = incomplete_label_forward(A, B, P, pi, observations[r], np.array(labels[r]))
        beta = incomplete_label_backward(A, B, P, observations[r], np.array(labels[r]), c)

        if update_A:
            # Calculate m_ij(l)
            m_ijl_r = calculate_m_ijl(A, B, P, observations[r], np.array(labels[r]), alpha, beta)
            m_ij_r = np.sum(m_ijl_r, axis=2)
            m_ij += m_ij_r

        if update_B or update_P:
            # Calculate m_i(l)
            m_il_r, m_ils_p_r, m_ils_m_r = calculate_m_il(alpha, alpha_p, alpha_m, beta, c_p, c_m)
            if update_B:
                m_ia += calculate_m_ia(m_il_r, B, observations[r])

            if update_P:
                # Calculate m_i(c)
                m_ic += calculate_m_ic(m_ils_p_r, m_ils_m_r, np.array(labels[r]))

    if update_A:
        #Update A according to equation B.8 given in [1]
        A = m_ij/(np.sum(m_ij, axis=1)).reshape(A.shape[0], 1)
    if update_B:
        # Update B according to equation B.7 given in [1]
        B = m_ia/(np.sum(m_ia, axis=1).reshape(B.shape[0], 1))
    if update_P:
        # Update P according to equation B.9 given in [1]
        P = m_ic/(np.sum(m_ic, axis=1).reshape(P.shape[0], 1))
    return A, B, P

def fit(A, B, P, pi, observations, labels, max_epoch, update_A=True, update_B=True, update_P=True):
    for epoch in range(max_epoch):
        A, B, P = update_matrices(A, B, P, pi, observations, labels, update_A, update_B, update_P)                                                                      
    return A, B, P
