"""
This code is written by Emil Vardar (student at KTH Royal Institute of Technology,
Stockholm, Sweden). And the goal is to update transition, emission and label distribution 
matrices for Hidden Markov Models (HMM) using conditional maximum likelihood approach (CML) 
where the labels for the states are 'incomplete'. In this file we DO NOT use normalization
in order to see whether the method given in [1] (see below) really is correct.Therefore,
the sequence length can only be very short in order to not get numerical issues. The 
normalized one is also uploaded with the name 'cml_incomplete_normalized.py'. 

It is open source. Use at your own risk.

The reference [1] refered many times below is:
[1]: Riis, Søren. Hidden Markov models and neural networks for speech recognition. Technical 
     University of Denmark [Department of Mathematical Modeling], 1998.
"""

import numpy as np
from numba import njit

@njit
def create_zeros(N, L, S):
    """
    Defines zero-matrices to be used in the forward-backward algorithm.
    """
    alpha = np.zeros((N, L, S+1))
    alpha_p = np.zeros_like(alpha)
    alpha_m = np.zeros_like(alpha)
    return alpha, alpha_p, alpha_m

@njit
def forward_initialization(B, P, pi, N, observations, labels, alpha, alpha_m, alpha_p):
    """
    This function does the initialization step given in Algorithm B.3 in [1].
    """
    for j in range(N):
        joint = B[j, observations[0]] * pi[j, 0]
        alpha_m[j, 0, 0] = joint * P[j, N]
        alpha_p[j, 0, 1] = joint * P[j, labels[0]] 
    alpha = alpha_m + alpha_p
    return alpha, alpha_m, alpha_p

@njit
def forward_recursion(L, S, N, A, B, P, observations, labels, alpha, alpha_p, alpha_m):
    """
    This function does the recursion step given in Algorithm B.3 in [1].
    """
    for l in range(1, L):
        for s in range(S+1):
            for j in range(N):
                temp_p = 0
                temp_m = 0
                for i in range(N):
                    a_ij = A[i, j]
                    if s != 0:
                        temp_p += a_ij * alpha[i, l-1, s-1]
                    temp_m += a_ij * alpha[i, l-1, s]
                b_jl = B[j, observations[l]]
                alpha_p[j, l, s] = b_jl * P[j, labels[s-1]] * temp_p   
                alpha_m[j, l, s] = b_jl * P[j, N] * temp_m 
                alpha[j, l, s] = alpha_p[j, l, s] + alpha_m[j, l, s]
    return alpha, alpha_p, alpha_m

@njit
def incomplete_label_forward(A, B, P, pi, observations, labels):
    """
    This function implements the forward algorithm given in Algorithm B.3 
    in [1]. This forward algorithm is for the special case where we 
    have incomplete labels (S<=L, where S is the number of labels and L 
    is the number of observaiton values). 
    """
    L = len(observations)    # Number of observations
    S = len(labels)          # Number of labels (S<=L) 
    N = A.shape[0]           # Number of states 

    # Create zero-matrices
    alpha, alpha_p, alpha_m = create_zeros(N, L, S)
    
    # Initialization step of Alg B.3 in [1]
    alpha, alpha_m, alpha_p = forward_initialization(B, P, pi, N, observations, labels, alpha, alpha_m, alpha_p)

    # Recursion step of Alg B.3 in [1]
    alpha, alpha_p, alpha_m = forward_recursion(L, S, N, A, B, P, observations, labels, alpha, alpha_p, alpha_m)

    # Return the updated values 
    return alpha, alpha_p, alpha_m

@njit
def backward_recursion(L, S, N, beta, A, B, P, labels, observations):
    """
    This function does the recursion step given in Algorithm B.4 in [1].
    """
    for l in range(L-2, -1, -1):
        for s in range(S, -1, -1):
            for i in range(N):
                temp_p = 0
                temp_m = 0
                for j in range(N):
                    if s != S:
                        temp_p += beta[j, l+1, s+1] * A[i, j] * P[j, labels[s]] * B[j, observations[l+1]] 
                    temp_m += beta[j, l+1, s] * A[i, j] * P[j, N] * B[j, observations[l+1]]
                beta[i, l, s] = temp_p + temp_m
    return beta

@njit
def incomplete_label_backward(A, B, P, observations, labels):
    """
    This function implements the backward algorithm given in Algorithm B.4 in [1] but 
    This backward algorithm is for the special case where we have incomplete labels (S<=L, 
    where S is the number of labels and L is the number of observaitons).
    """
    L = len(observations)    # Number of observations
    S = len(labels)          # Number of labels S<=L. 
    N = A.shape[0]           # Number of states

    beta = np.zeros((N, L, S+1))

    # Initializaiton step given in algorithm B.4 in [1]
    beta[:, L-1, S] = 1
    
    # Recursion step
    beta = backward_recursion(L, S, N, beta, A, B, P, labels, observations)
    return beta

@njit
def calculate_m_il(alpha_norm, alpha, alpha_p, alpha_m, beta):
    """
    Calculates the m_i+(l, s) and m_i-(l, s) according to equation B.21 and B.22
    in [1]. Furthermore, calculates the m_i(l) by summing these two according to
    equation B.23.
    """      
    N = alpha.shape[0]    # Number of states
    L = alpha.shape[1]    # Number of observations
    S = alpha.shape[2]-1  # Number of labels S<=L. 
    
    m_ils_p = np.zeros_like(alpha)    # The positive part of m_i(l, s) --> m_i+(l,s)
    m_ils_m = np.zeros_like(alpha)    # The negative part of m_i(l, s) --> m_i-(l,s)
    
    denom = np.sum(np.sum(alpha_norm*beta, axis=2), axis=0)
    for i in range(N):
        for l in range(L):
            for s in range(S+1):
                m_ils_p[i, l, s] = alpha_p[i, l, s] * beta[i, l, s] / denom[l] 
                m_ils_m[i, l, s] = alpha_m[i, l, s] * beta[i, l, s] / denom[l]

    # Sum m_i+(l, s) and m_i-(l, s) 
    m_il = np.sum((m_ils_p + m_ils_m), axis=2)

    gamma = np.zeros(N)
    for i in range(N):
        temp = 0
        for s in range(S+1):
            temp += alpha[i,0,s] * beta[i,0,s] / denom[0]
        gamma[i] = temp

    return m_il, m_ils_p, m_ils_m, gamma

@njit
def calculate_m_ijl(A, B, P, observations, labels, alpha, beta):
    """
    Calculates m_ij(l) according to equation B.24 in [1].
    """  
    N = alpha.shape[0]    # Number of states
    L = alpha.shape[1]    # Number of observations
    S = alpha.shape[2]-1  # Number of labels S<=L. 

    m_ijl = np.zeros((N, N, L-1))
    denom = np.sum(np.sum(alpha*beta, axis=2), axis=0)

    # Then calculate the numerator and divide with correct denominator calcualted above
    for i in range(N):
        for j in range(N):
            for l in range(1, L):
                temp_p = 0
                temp_m = 0
                for s in range(S+1):
                    if s != 0:
                        temp_p += alpha[i, l-1, s-1] * B[j, observations[l]] * P[j, labels[s-1]] * A[i, j] * beta[j, l, s]
                    temp_m += alpha[i, l-1, s] * B[j, observations[l]] * P[j, -1] * A[i, j] * beta[j, l, s] 
                m_ijl[i, j, l-1] = (temp_p + temp_m) / (denom[l])
    return m_ijl

@njit
def calculate_m_ic(m_ils_p, m_ils_m, labels):
    """
    Calculates m_i(c) according to equation B.26 in [1]
    """
    N = m_ils_p.shape[0]    # Number of states
    L = m_ils_p.shape[1]    # Number of observations
    S = m_ils_p.shape[2]-1  # Number of labels S<L. 
    
    m_ic = np.zeros((N, N+1))

    for c in range(N+1):
        for i in range(N):
            temp = 0
            if c < N:
                for l in range(L):
                    for s in range(1, S+1):
                        if c == labels[s-1]:
                            temp += m_ils_p[i, l, s]
                m_ic[i, c] = temp
            if c == N:
                for l in range(L):
                    for s in range(1, S+1):  # Seems to work better with the range [1, S+1) but I am not sure 
                        temp += m_ils_m[i, l, s]
                m_ic[i, c] = temp
    return m_ic

@njit
def calculate_m_ia(m_il, B, observations):
    """
    Calculates m_i(a) from m_i(l) according to equation B.7 in [1]. 
    """
    m_ia = np.zeros_like(B)
    for i in range(B.shape[0]):
        for a in range(B.shape[1]):
            temp = 0
            for l in range(m_il.shape[1]):
                if observations[l] == a:
                    temp += m_il[i, l]
            m_ia[i, a] = temp 
    return m_ia

def update_1_epoch(A, B, P, pi, observations, labels, update_A, update_B, update_P, update_pi):
    """
    Goes through the training set onces and updates A, B, and P matrices according to the
    training set. 
    """
    m_ij = np.zeros_like(A)
    m_ia = np.zeros_like(B)
    m_ic = np.zeros_like(P)
    gamma_total = np.zeros(A.shape[0])
    
    ln_P_list = []
    for r in range(len(observations)):
        # Calculate the necessary forward and backward variables
        alpha, alpha_p, alpha_m = incomplete_label_forward(A, B, P, pi, observations[r], np.array(labels[r]))
        beta = incomplete_label_backward(A, B, P, observations[r], np.array(labels[r]))
        ln_P_list.append(np.log(np.sum(alpha[:,7,:] * beta[:,7,:])))
        
        if update_A:
            # Calculate m_ij(l)
            m_ijl_r = calculate_m_ijl(A, B, P, observations[r], np.array(labels[r]), alpha, beta)
            m_ij_r = np.sum(m_ijl_r, axis=2)
            m_ij += m_ij_r

        if update_B or update_P or update_pi:
            # Calculate m_i(l)
            m_il_r, m_ils_p_r, m_ils_m_r, gamma = calculate_m_il(alpha, alpha, alpha_p, alpha_m, beta)
            if update_B:
                m_ia += calculate_m_ia(m_il_r, B, observations[r])

            if update_P:
                # Calculate m_i(c)
                m_ic += calculate_m_ic(m_ils_p_r, m_ils_m_r, np.array(labels[r]))
            
            if update_pi:
                gamma_total += gamma
    
    if update_A:
        #Update A according to equation B.8 given in [1]
        A = m_ij/(np.sum(m_ij, axis=1)).reshape(A.shape[0], 1)
    if update_B:
        # Update B according to equation B.7 given in [1]
        B = m_ia/(np.sum(m_ia, axis=1).reshape(B.shape[0], 1))
    if update_P:
        # Update P according to equation B.9 given in [1]
        P = m_ic/(np.sum(m_ic, axis=1).reshape(P.shape[0], 1))
    if update_pi:
            # Update pi
        pi = gamma_total/(np.sum(gamma_total))
        pi = pi.reshape((len(pi), 1))
    return A, B, P, pi, np.sum(ln_P_list)/len(ln_P_list)

def fit(A, B, P, pi, observations, labels, max_epoch, update_A=True, update_B=True, update_P=True, update_pi=True):
    """
    Goes through the training set max_epoch times and updates the A, B, and P matrices. Retruns the 
    updated matrices. 
    """
    ln_P_list = []
    for epoch in range(max_epoch):
        A, B, P, pi, ln_P = update_1_epoch(A, B, P, pi, observations, labels, update_A, update_B, update_P, update_pi)                                                    
        ln_P_list.append(ln_P)
    return A, B, P, pi, ln_P_list
