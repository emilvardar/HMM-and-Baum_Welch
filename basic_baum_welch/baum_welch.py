"""
This code is written by Emil Vardar, and it is open-source. Use it on your own risk

This code is implements the forward-backward and the Baum-Welch algorithms
based on the normalized alpha and beta parameters in order to counter act
the numerical issues. 

Reference cited many times below is 
[1]: Shen, Dawei. "Some mathematics for HMM." Massachusetts Institute of Technology (2008).
"""

import numpy as np
from numba import njit

@njit
def forward(A, B, pi, O):
    """
    Implements the forward algorithm according to [1].
    
    Inputs:
    A: The transition matrix of shape (N, N) where N is # of states
    B: The emission matrix of shape (N, K) where K is number of possible outcomes
    pi: The initial state distributions of shape (N, 1)
    O: The observations in list form
    
    Returns:
    ln_P: Total observation probability
    alpha_hat: Scaled forward variable of shape (N, T) where T is the length of observations
    c: Forward scale factors of shape (T,) 
    """
    
    T = len(O)             # Length of the observations
    N = A.shape[0]         # Number of states
    
    alpha_2dot = np.zeros((N, T))
    alpha_hat = np.zeros_like(alpha_2dot)
    c = np.zeros(T)
    
    # Implement the initialization procedure according to equations 10 given in [1]
    alpha_2dot[:,0] =  pi[:,0]*B[:, O[0]]       
    c[0] = 1/np.sum(alpha_2dot[:,0])            
    alpha_hat[:,0] = c[0] * alpha_2dot[:,0]   
    
    # Implement the induction procedure according to equations 11 given in [1]
    for t in range(1, T):
        for i in range(N):
            summation = 0
            for j in range(N):
                summation += alpha_hat[j, t-1] * A[j, i]     
            alpha_2dot[i, t] = summation * B[i, O[t]]        
        c[t] = 1/(np.sum(alpha_2dot[:,t]))                   
        alpha_hat[:, t] = c[t] * alpha_2dot[:,t] 

    ln_P = -np.sum(np.log(c))     # Eq. 14 given in [1]
    return ln_P, alpha_hat, c

@njit
def backward(A, B, O, c):
    """
    Implements the backward algorithm according to [1].
    
    Inputs:
    A: The transition matrix of shape (N, N) where N is # of states
    B: The emission matrix of shape (N, K) where K is number of possible outcomes
    O: The observations in list form
    c: Forward scale factors of shape (T,) 
    
    Returns:
    beta_hat: Scaled backward variable of shape (N, T) where T is the length of observations
    """
    T = len(O)             # Length of the observations
    N = A.shape[0]         # Number of states
    
    beta_hat = np.zeros((N, T))
    
    # Implement the initialization procedure according to equation 15 given in [1]
    beta_hat[:, -1] = c[-1]
    
    # Implement the induction procedure according to equation 16 given in [1]
    for t in range(T-2, -1, -1):
        for i in range(N):
            summation = 0
            for j in range(N):
                summation += A[i, j] * B[j, O[t+1]] * beta_hat[j, t+1]  
            beta_hat[i, t] = summation * c[t]
    return beta_hat

@njit
def calculate_num_A(N, T, alpha_hat_l, beta_hat_l, c_l, A, B, observation_l, num_A, gamma, l, hold_A, hold_pi):
    """
    Calculates first summation (from t=1 to T-1) of the numerator for the update of transition matrix
    A given in equation 24 in [1]. It also calculates the gamma for updating the initial distribution
    matrix.
    
    Inputs:
    N: Number of states
    T: Length of the observation sequence
    alpha_hat_l: Scaled forward variable for the l:th training example
    beta_hat_l: Scaled backward variable for the l:th training example
    c_l: Forward scale factors for the l:th training example
    A: transition matrix
    B: emission matrix
    observation_l: the l:th observation sequence 
    num_A: Array holding the calculated numerator values for A
    gamma: Update variable for the initial distributions
    l: the number of the training sequence
    hold_x: Booleans denoting if A and pi should be updated or not respectively
    """
    for i in range(N):
        if not hold_A:
            for j in range(N):
                num_sum = 0
                for t in range(T-1):
                    num_sum += alpha_hat_l[i, t] * A[i,j] * B[j, observation_l[t+1]] * beta_hat_l[j, t+1]
                num_A[l, i, j] = num_sum
        if not hold_pi:
            gamma[l, i] = alpha_hat_l[i, 0] * beta_hat_l[i, 0] / c_l[0]

@njit
def calculate_num_denom_B(K, N, T, observation_l, alpha_hat_l, beta_hat_l, c_l, num_B, denom_B, l):
    """
    Calculates the first summation (from t=1 to T-1) of the numerator and denominator for the 
    update of emission matrix according to equation 24 given in [1]. 
    
    Inputs:
    K: Number of possible outcomes
    N: Number of states
    T: Length of the observation sequence
    observation_l: the l:th observation sequence 
    alpha_hat_l: Scaled forward variable for the l:th training example
    beta_hat_l: Scaled backward variable for the l:th training example
    c_l: Forward scale factors for the l:th training example
    num_B: Array holding the calculated numerator values for B
    denom_B: Array holding the calculated denominator values for B
    l: the number of the training sequence
    """
    for k in range(K):
        for j in range(N):
            numerator = 0
            denominator = 0
            for t in range(T):
                if observation_l[t] == k:
                    numerator += (alpha_hat_l[j, t] * beta_hat_l[j, t]) / c_l[t]
                denominator += (alpha_hat_l[j, t] * beta_hat_l[j, t]) / c_l[t] 
            num_B[l, j, k] = numerator
            denom_B[l, j, k] = denominator

@njit
def calculate_denom_A(N, T, alpha_hat_l, beta_hat_l, c_l, l, denom_A):
    for i in range(N):
        denominator = 0
        for t in range(T-1):
            denominator += (alpha_hat_l[i, t] * beta_hat_l[i, t]) / c_l[t] 
        denom_A[l, i] = denominator

@njit
def bw(A, B, pi, O, hold_A, hold_B, hold_pi):
    """
    Does the Baum-welch iteration according to [1]
    
    Inputs:
    A: The transition matrix of shape (N, N) where N is # of states
    B: The emission matrix of shape (N, K) where K is number of possible outcomes
    pi: The initial state distributions of shape (N, 1)
    O: The observations in list form
    hold_X: Booleans denoting if A, B, and pi should be updated or not respectively

    Ouputs:
    A: Updated A
    B: Updated B
    pi: Updated pi
    likelihood_of_sequence: The mean total observation probability
    """

    N = A.shape[0]             # Num of states
    K = B.shape[1]             # Num of possible outcomes 
    num_sequence = O.shape[0]  # Num of training sequences
    likelihood_of_sequence_holder = np.zeros(num_sequence)
 
    num_A = np.zeros((num_sequence, N, N))
    denom_A = np.zeros((num_sequence, N))
    num_B = np.zeros((num_sequence, N, K))
    denom_B = np.zeros_like(num_B)
    gamma = np.zeros((num_sequence, N))

    for l in range(num_sequence):       # Go through all the training sequences
        observation_l = O[l]            # The l:th training sequence
        T = len(observation_l)

        likelihood_of_sequence_holder[l], alpha_hat_l, c_l = forward(A, B, pi, observation_l)
        beta_hat_l = backward(A, B, observation_l, c_l)  
        
        calculate_num_A(N, T, alpha_hat_l, beta_hat_l, c_l, A, B, observation_l, num_A, gamma, l, hold_A, hold_pi) 
        if not hold_B:  
            calculate_num_denom_B(K, N, T, observation_l, alpha_hat_l, beta_hat_l, c_l, num_B, denom_B, l)
        if not hold_A:
            calculate_denom_A(N, T, alpha_hat_l, beta_hat_l, c_l, l, denom_A)

    # After going through all the training examples, we can now update A, B and pi 
    # matrices according to equation 24 given in [1].
    if not hold_A:
        numerator_A = np.sum(num_A, axis = 0)
        denominator_A = np.sum(denom_A, axis=0).reshape(A.shape[0], 1)
        A = numerator_A/denominator_A

    if not hold_B:
        numerator_B = np.sum(num_B, axis = 0)
        denominator_B = np.sum(denom_B, axis = 0)
        B = numerator_B / denominator_B

    if not hold_pi:
        gamma_j = np.sum(gamma, axis = 0)
        pi = gamma_j/(np.sum(gamma_j))
        pi = pi.reshape((len(pi), 1))

    # The mean value of the likelihood of the observed sequence
    likelihood_of_sequence = np.sum(likelihood_of_sequence_holder)/num_sequence 
    return A, B, pi, likelihood_of_sequence


def fit(A, B, O, N, TOL = 1e-3, NUM_ITER = 100, pi = 0, hold_A=False, hold_B=False, hold_pi=False):
    """
    Calls the baum-welch algorithm NUM_ITER times.
    A: the transition matrix of shape NxN given as a numpy array
    B: the emission matrix of shape NxK given as a numpy array
    O: the observations given of shape number_of_sequences x number of observations per sequence. Is a numpy array.
    N: number of states
    TOL: tolerance for stopping the iteration
    NUM_ITER: maximum number of iterations 
    pi: the initial transition matrix. If it is not given as input all the states are seen as equal probability.  
    """

    delta = 10
    likelihood_list = []
    if type(pi) == int:  # if pi is not given start with equal probability on all the entries
        pi = np.array((np.ones(N)/N).reshape((-1, N))).T
    
    iter = 0
    while delta>TOL and iter<NUM_ITER:
        A, B, pi, prob_for_comp = bw(A, B, pi, O, hold_A, hold_B, hold_pi)
        likelihood_list.append(prob_for_comp)
        try:
            delta = likelihood_list[-1] - likelihood_list[-2]
        except:
            pass
        iter += 1
    return A, B, pi, likelihood_list, iter, delta

