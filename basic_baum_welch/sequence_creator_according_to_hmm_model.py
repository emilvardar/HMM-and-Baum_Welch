import numpy as np
import bisect

def sequence_deciders(A_gt, B_gt, pi_gt, K, L):
    '''The pure reason for this function is to create lists in order to be able to take steps according to markov chain.
    We define the lists according to each state transition and from state to observation transition according to the given
    markov process.'''
    sequence_decider_A = {}
    for k in range(K):
        sequence_decider_A[k] = []
        for k2 in range(K-1):
            try:
                sequence_decider_A[k].append(A_gt[k, k2] + sequence_decider_A[k][-1]) 
            except:
                sequence_decider_A[k].append(A_gt[k, k2]) 
        sequence_decider_A[k].append(1.0)   # Need to do this because of rounding error that occurs in Python


    sequence_decider_B = {}
    for k in range(K):
        sequence_decider_B[k] = []
        for l in range(L-1):
            try:
                sequence_decider_B[k].append(B_gt[k, l] + sequence_decider_B[k][-1]) 
            except:
                sequence_decider_B[k].append(B_gt[k, l])
        sequence_decider_B[k].append(1.0) # Need to do this because of rounding error that occurs in Python

    sequence_decider_pi = []
    sequence_decider_pi.append(pi_gt[0])
    for i in range(1, pi_gt.shape[0]):
        sequence_decider_pi.append(sequence_decider_pi[-1] + pi_gt[i])
    
    return sequence_decider_A, sequence_decider_B, sequence_decider_pi

def create_sequences(A_gt, B_gt, pi_gt, K, T, N, L):
    'This function creates seqeunces according to the markov process defined above.'
    A_decider, B_decider, pi_decider = sequence_deciders(A_gt, B_gt, pi_gt, K, L)
    states_list = []
    observations_list = []

    for num_seq in range(T):
        states = []
        observations = []
        rand_num = np.random.uniform()
        states.append(bisect.bisect_left(pi_decider, rand_num))
        rand_num = np.random.uniform()
        observations.append(bisect.bisect_left(B_decider[states[-1]], rand_num))
        for i in range(N-1):
            rand_num = np.random.uniform()
            states.append(bisect.bisect_left(A_decider[states[-1]], rand_num))
            rand_num = np.random.uniform()
            observations.append(bisect.bisect_left(B_decider[states[-1]], rand_num))
        states_list.append(states)
        observations_list.append(observations)
    return states_list, observations_list


def create_gt(K, L):
    # A is a matrix of KxK. The indicies are such that a_ij is the prob of 
    # going from state i to state j
    A_gt = np.random.rand(K,K)
    A_gt = np.divide(A_gt, np.sum(A_gt, axis=0)).T
    A_gt = np.round(A_gt, 3)

    # B is a matrix of KxL w,th similar build-up as A. b_ij indices the prob
    # of going from state i to observations j.
    B_gt = np.random.rand(L,K)
    B_gt = np.divide(B_gt, np.sum(B_gt, axis=0)).T
    B_gt = np.round(B_gt, 3)

    # pi is a matrix of Kx1. 
    pi_gt = np.random.rand(K,1)
    pi_gt = np.divide(pi_gt, np.sum(pi_gt, axis=0))
    pi_gt = np.round(pi_gt, 3)
    return A_gt, B_gt, pi_gt 


def create_gt_edhmm(K, L, S, D):
    # A is a matrix of KxK. The indicies are such that a_ij is the prob of 
    # going from state i to state j
    A_gt = np.zeros((K,K))
    for k in range(K):
        if k % D == D-1:
            random_array = np.abs(np.random.randn(K))
            A_gt[k, :] = random_array/np.sum(random_array)
        elif k % D == 0:
            rand_num = np.random.uniform()
            while rand_num > 0.2:
                rand_num = np.random.uniform()
            A_gt[k, k] = rand_num
            A_gt[k, k+1] = 1-rand_num
        else:
            A_gt[k, k+1] = 1        
    A_gt = A_gt
    
    # B is a matrix of KxL w,th similar build-up as A. b_ij indices the prob
    # of going from state i to observations j.
    B_gt = np.zeros((K, L))
    start_num = 0
    random_array = np.abs(np.random.randn(L))
    random_array = random_array/np.sum(random_array)
    for k in range(K):    
        if k//D == start_num:
            B_gt[k, :] = random_array
        else:
            start_num = k//D
            random_array = np.abs(np.random.randn(L))
            random_array = random_array/np.sum(random_array)
            B_gt[k, :] = random_array
    B_gt = B_gt
            
    # pi is a matrix of Kx1. 
    pi_gt = np.random.rand(K,1)
    pi_gt = np.round(np.divide(pi_gt, np.sum(pi_gt, axis=0)), 3)
    for k in range(K):
        if k % D == 0:
            while pi_gt[k] > 0.10:
                pi_gt[k] = np.random.uniform()
    pi_gt = np.divide(pi_gt, np.sum(pi_gt, axis=0))
    return A_gt, B_gt, pi_gt 


def hmmgenerate(K, L, T, N, S=0, D=0, edhmm = False):
    if not edhmm: 
        A_gt, B_gt, pi_gt = create_gt(K, L)
    if edhmm:
        A_gt, B_gt, pi_gt = create_gt_edhmm(K, L, S, D)
    teoretical_states, teoretical_observations = create_sequences(A_gt, B_gt, pi_gt, K, T, N, L)
    teoretical_observations = np.array(teoretical_observations)
    return A_gt, B_gt, pi_gt, teoretical_observations, teoretical_states

if __name__ == '__main__':
    A_gt, B_gt, pi_gt, teoretical_observations, teoretical_states = hmmgenerate(8, 16, 100, 4096, S=0, D=0, edhmm = False)
