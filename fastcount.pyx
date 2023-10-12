# from cpython cimport array
# import array
import numpy as np
cimport numpy as np


def transitions(n, trails):
    n_samples, _ = trails.shape
    cdef np.ndarray c = np.zeros([n_samples, n, n], dtype=int)
    for t, trail in enumerate(trails):
        i = trail[0]
        for j in trail[1:]:
            c[t, i, j] += 1
            i = j
    return c


def avg_transition_time(n, trails):
    n_samples, _ = trails.shape
    cdef np.ndarray sum_ttime = np.zeros([n_samples, n], dtype=int)
    cdef np.ndarray num_t = np.zeros([n_samples, n], dtype=int)
    for t, trail in enumerate(trails):
        l = 0
        i = trail[0]
        for j in trail[1:]:
            l += 1
            if i != j:
                sum_ttime[t, i] += l
                num_t[t, i] += 1
                l = 0
            i = j
    return sum_ttime, num_t


def continuous_time_likelihood(mixture, trails_ct):
    lls = np.zeros((mixture.L, len(trails_ct)))
    for l in range(mixture.L):
        for trail_ix, trail in enumerate(trails_ct):
            first_state = trail[0][0]
            ll = np.log(mixture.S[l, first_state] + 1e-10)
            transition_time = 0
            for (i, time), (j, _) in zip(trail[:-1], trail[1:]):
                transition_time += time
                if i != j:
                    lam = mixture.Ks[l, i, j] + 1e-8
                    ll += np.log(lam * np.exp(-lam * transition_time))
                    transition_time = 0
            lls[l, trail_ix] = ll
    return lls
