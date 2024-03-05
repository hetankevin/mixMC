import numpy as np
import scipy
import itertools
import fastcount
import time

import dtlearn as dt
from ctmixtures import *



def infinitesimal_single_chain(n, trails, trail_weights=None, tau=1, return_counts=False):
    """learn a CTMC from trails, assuming tau -> 0
    
    n: number of states
    trails: array of trails, shape (num_trails, t_len) where t_len is the number of state observations per trail
    trail_weights: weight of each trail, used for soft EM
    tau: step duration
    """
    if trail_weights is None: trail_weights = np.ones(len(trails))

    C_start = np.sum(trail_weights[:, None] * (trails[:, 0, None] == np.arange(n)[None, :]), axis=0)
    C_start = np.clip(C_start, 1e-10, np.inf)
    S = C_start / np.sum(C_start)

    T = np.zeros((n, n))
    C = np.zeros((n, n))
    for i, j in np.ndindex(n, n):
        C[i, j] = np.sum(trail_weights * np.sum((trails[:, :-1] == i) * (trails[:, 1:] == j), axis=1))
    C = np.clip(C, 1e-10, np.inf)
    T = C / np.sum(C, axis=1)[:, None]
    K = (T - np.eye(n)) / tau
    assert(not np.isnan(K).any()), f"NaN in K"

    mixture = Mixture([S], [K])
    if return_counts: return mixture, C
    else: return mixture


def mle_single_chain(n, trails, trail_weights=None, tau=1, eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=1000, conv_thresh=1e-5, verbose=False):
    """Learns a CTMC from trails, iteratively refining the MLE using ADAM (cf. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4514821/)
    Output is a tuple (S, K) where S are the starting probabilities and K is the rate matrix

    n: number of states
    trails: array of trails, shape (num_trails, t_len) where t_len is the number of state observations per trail
    trail_weights: weight of each trail, used for soft EM
    tau: step duration
    """
    mixture, C = infinitesimal_single_chain(n, trails, trail_weights=trail_weights, tau=tau, return_counts=True)

    def ll(K):
        matrix_exp = scipy.linalg.expm(K * tau)
        matrix_exp = np.clip(matrix_exp, 1e-10, 1e10)
        return np.sum(C * np.log(matrix_exp))

    def grad_num(K):
        G = np.zeros_like(K)
        ll0 = ll(K)
        for i, j in itertools.product(*map(range, K.shape)):
            if i == j: continue
            D = np.zeros_like(K)
            D[i, j] = eps
            D[i, i] = -eps
            G[i, j] = (ll(K + D) - ll0) / eps
        G[range(n), range(n)] = -np.sum(G, axis=1)
        assert(not np.isnan(G).any()), "NaN in numerical differentiation"
        return G

    def grad(K):
        try:
            lam, V = np.linalg.eig(K)
            U = np.linalg.inv(V).T
            lam += np.random.random(lam.shape) * 1e-10
            X_num = np.exp(tau*lam)[:,None] - np.exp(tau*lam)[None,:]
            X_num[range(n), range(n)] = tau * np.exp(tau*lam)
            X_den = lam[:,None] - lam[None,:]
            X_den[range(n), range(n)] = 1
            X = X_num / X_den
            T = scipy.linalg.expm(K * tau)
            D = C / T
            Z = U @ ((V.T @ D @ U) * X) @ V.T
            dL = Z - np.diag(Z)[:,None]
            dL[range(n), range(n)] = -np.sum(dL, axis=1)
            if np.isnan(dL).any(): raise np.linalg.LinAlgError
            return np.real(dL)
        except np.linalg.LinAlgError:
            return grad_num(K)

    ll_best = None
    K_best = None
    K_prev = None

    def eval_ll():
        if verbose and i % 100 == 0: print(f"Iteration {i}: ...")
        K = mixture.Ks[0]
        nonlocal ll_best, K_best, K_prev
        assert(not np.isnan(K).any()), "NaN in K"
        # if not np.isnan(K).any(): return True
        ll_now = ll(K)
        if ll_best is None or ll_now > ll_best:
            ll_best = ll_now
            K_best = K
        linf = 1 if K_prev is None else np.max(np.abs(K - K_prev))
        K_prev = K
        if verbose and i % 100 == 0: print(f"Iteration {i}: ll={ll_now:.4f} linf={linf:.4f}")
        return linf < conv_thresh

    # mixture.Ks += np.random.random(mixture.Ks[0].shape) * 1e-5
    # mixture.normalize()
    m = np.zeros_like(mixture.Ks[0])
    v = np.zeros_like(mixture.Ks[0])
    for i in range(max_iter or 1000):
        if eval_ll(): break

        g = grad(mixture.Ks[0])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dK = eta * m_hat / (np.sqrt(v_hat) + eps)

        mixture.Ks[0] += dK
        mixture.normalize()

    eval_ll()
    mixture.Ks[0] = K_best
    mixture.normalize()
    return mixture


def mle_prior(lls, *args, **kwargs):
    chains = [mle_single_chain(*args, **kwargs, trail_weights=ll) for ll in lls]
    return Mixture.from_combination(chains)



# def mixture_dt2ct(mixture, trails, tau=1, soft=True):
#     """Converts a discrete-time mixture to a continuous-time mixture.
#     
#     mixture: approximation to T(tau), will be used to cluster trails
#     """
#
#     """
#     _, t_len = trails.shape
#     n = mixture.n
#     L = mixture.L
#
#     logS = np.log(mixture.S + 1e-10)
#     logMs = np.log(mixture.Ms)
#
#     logl = logS[:, trails[:,0]]
#     for i in range(1, t_len):
#         logl += logMs[:, trails[:,i-1], trails[:,i]]
#     ll = np.exp(logl - np.max(logl, axis=0))
#     ll /= np.sum(ll, axis=0)[None, :]
#     """
#     n = mixture.n
#     L = mixture.L
#
#     ll = dt.likelihood(mixture, trails)
#
#     if soft:
#         # weights = np.exp(ll - np.max(ll, axis=1)[:, None])
#         # weights /= np.sum(weights, axis=1)[:, None]
#         chains = [mle(n, trails, trail_weights=ll[i], tau=tau) for i in range(L)]
#
#     else:
#         labels = np.argmax(ll, axis=0)
#         clustered_trails = [trails[labels == i] for i in range(L)]
#         chains = [mle(n, chain_trails, tau=tau) for chain_trails in clustered_trails]
#
#     mixture = CTMixture.from_combination(chains)
#     mixture.normalize()
#     return mixture


def svd_learn(n, L, trails, tau=1, mle_max_iter=None):
    """Learns a continuous-time mixture by applying an SVD-method to learn T(tau) first.

    n: number of states
    L: number of components
    trails: array of trails, shape (num_trails, 3) where each trail consists of 3 state observations
    soft: whether to use soft Mcgibbon
    """
    num_trails, t_len = trails.shape
    # assert(t_len == 3) # this can be alleviated by considering sub-trails of length 3 (by jumping over states), but using the full trail for MLE

    sample = dt.Distribution.from_trails(n, trails)
    mixture_dt = dt.svd_learn(sample, n, L)

    lls = dt.likelihood(mixture_dt, trails)
    return mle_prior(lls, n, trails, tau=tau, max_iter=mle_max_iter)

#   flat_mixture = mixture.flat()
#   flat_trails, _ = sample.flat_trails()
#
#   lls = flat_trails @ np.log(flat_mixture + 1e-20).transpose()
#
#   if soft:
#       weights = np.exp(lls - np.max(lls, axis=1)[:, None])
#       weights /= np.sum(weights, axis=1)[:, None]
#       chains = [mle(n, trails, trail_weights=weights[:, i], tau=tau) for i in range(L)]
#       """
#       weights = np.exp(lls - np.max(lls, axis=1)[:, None])
#       weights /= np.sum(weights, axis=1)[:, None]
#       chain_trails = [trails[np.random.sample(num_trails) < weights[:, i]] for i in range(L)]
#       chains = [mle(n, trails, tau=tau) for trails in chain_trails]
#       """
#
#   else:
#       labels = np.argmax(lls, axis=1)
#       clustered_trails = [trails[labels == i] for i in range(L)]
#       chains = [mle(n, chain_trails, tau=tau) for chain_trails in clustered_trails]
#
#   mixture = CTMixture.from_combination(chains)
#   mixture.normalize()
#   return mixture



def kausik_learn(n, L, trails, tau, return_labels=False, return_time=False, mle_max_iter=None):
    """Learns a continuous-time mixture by applying Kausik's method to cluster."""
    labels = dt.kausik_cluster(n, L, trails)
    lls = labels[None, :] == np.arange(L)[:, None]
    mle_start_time = time.time()
    mixture_ct = mle_prior(lls, n, trails, tau=tau, max_iter=mle_max_iter)
    mle_time = time.time() - mle_start_time
    if return_labels or return_time:
        return (mixture_ct,) + ((labels,) if return_labels else ()) + ((mle_time,) if return_time else ())
    else:
        return mixture_ct
    # return (mixture_ct, labels) if return_labels else mixture_ct

#   clustered_trails = dt.kausik_learn(n, *args, **kwargs, learn=False)
#   X = [mle(n, chain_trails, tau=tau) for chain_trails in clustered_trails]
#   S, Ks = map(np.array, zip(*X))
#   mixture = CTMixture(S, Ks)
#   mixture.normalize()
#   return mixture


def em_learn(n, L, trails, tau=1, n_iter=100, init=None):
    """EM algorithm for learning a continuous-time mixture from trails.
    Uses continuous-time MLE (McGibbons et al.) in every step.
    
    n: number of states
    L: number of chains
    trails: n_samples x t_len array of trails
    """
    n_samples, t_len = trails.shape
    mixture = Mixture.random(n, L) if init is None else init
    # eps = 1e-10

    for _ in range(n_iter):
        mixture_dt = mixture.toDTMixture(tau)
        lls = dt.likelihood(mixture_dt, trails)
        mixture = mle_prior(lls, n, L, trails, tau=tau)

        """
        logS = np.log(mixture.S + eps)
        logTs = np.log(mixture.Ts(tau))

        logl = logS[:, trails[:,0]] # L x n_samples
        for i in range(1, t_len):
            logl += logTs[:, trails[:,i-1], trails[:,i]]
        probs = np.exp(logl - np.max(logl, axis=0))
        probs /= np.sum(probs, axis=0)[None, :]

        chains = [mle(n, trails, trail_weights=probs[i], tau=tau) for i in range(L)]
        mixture = CTMixture.from_combination(chains)
        mixture.normalize()
        """

    return mixture


"""
def em_learn_ct_from_dt(n, L, trails, tau=1, soft=True):
    # mixture_dt = em_long_trails(n, L, trails, n_iter=n_iter)
    mixture_dt = em_learn2(n, L, trails)
    mixture_ct = mixture_dt2ct(mixture_dt, trails, tau=tau, soft=soft)
    return mixture_ct
"""


"""
def em_learn_rec(n, L, trails, tau=1):
    _, t_len = trails.shape
    num_steps = 3
    f = 2

    mixture = CTMixture.random(n, L)
    while True:
        stepsize = int(np.ceil(t_len / num_steps))
        trails_ = trails[:, ::stepsize]
        mixture_dt = mixture.toDTMixture(tau * stepsize)
        mixture_dt = em_learn2(n, L, trails_, init=mixture_dt)
        mixture = mixture_dt2ct(mixture_dt, trails, tau=tau * stepsize)
        if stepsize == 1: break
        mixture.Ks /= f
        num_steps *= f
    
    return mixture
"""

def em_learn_init(n, L, trails, tau=1):
    rates = em_learn_rates(n, L, trails, tau=tau)
    mixture = Mixture.random(n, L)
    mixture.Ks = np.array([- K / np.diag(K)[:, None] * r[:, None] for K, r in zip(mixture.Ks, rates)])
    return mixture


def em_learn_rates(n, L, trails, tau=1, max_iter=100, conv_thresh=1e-5, verbose=False):
    sum_ttime, num_t = fastcount.avg_transition_time(n, trails)
    sum_ttime = tau * sum_ttime.T[None,:,:] # chain x state x trail
    num_t = num_t.T[None,:,:]
    rates = np.random.random((L, n, 1)) / 100
    prev_ll = 0
    
    for iter in range(max_iter):
        log_ll = np.sum(num_t * np.log(rates) - rates * sum_ttime, axis=1)
        ll = np.exp(log_ll - np.max(log_ll, axis=0))
        ll /= np.sum(ll, axis=0)[None, :]
        ll = ll[:, None, :]

        avg_ttime = np.sum(ll * sum_ttime, axis=2) / np.sum(ll * num_t, axis=2)
        avg_ttime = avg_ttime[:, :, None]
        rates = 1 / avg_ttime

        linf = np.max(np.abs(ll - prev_ll))
        if verbose: print(f"Iteration {iter+1}/{max_iter}: {linf}")
        if linf < conv_thresh: break
        prev_ll = ll

    return rates[:, :, 0]

def soft_clustering(mixture, trails_ct):
    log_ll = likelihood(mixture, trails_ct)
    lls = np.exp(log_ll - np.max(log_ll))
    lls /= np.sum(lls, axis=0)

    ixs = np.any(np.isnan(lls), axis=0)
    lls[:, ixs] = 0

    return lls

def likelihood(mixture, trails_ct):
    return fastcount.continuous_time_likelihood(mixture, trails_ct)

    """
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
    """


def mle_ct_single_chain(n, trails, trail_weights=None):
    if trail_weights is None: trail_weights = np.ones(len(trails))
    holding_times = [[1] for _ in range(n)]
    holding_time_weights = [[1e-10] for _ in range(n)]
    n_transitions = np.ones((n, n)) * 1e-10
    for trail, w in zip(trails, trail_weights):
        for (i, time), (j, _) in zip(trail[:-1], trail[1:]):
            holding_times[i].append(time)
            holding_time_weights[i].append(w)
            n_transitions[i, j] += w

    holding_diag = np.array([1 / np.average(ts, weights=ws) for ts, ws in zip(holding_times, holding_time_weights)])
    transition_prob = n_transitions / np.sum(n_transitions, axis=1)[:, None]
    transition_prob[range(n), range(n)] = -1

    K = holding_diag[:, None] * transition_prob
    starting_states, starting_state_counts = np.unique([trail[0][0] for trail in trails], return_counts=True)
    S = np.zeros(n)
    S[starting_states] = starting_state_counts / np.sum(starting_state_counts)

    return Mixture([S], [K])

def mle_ct_prior(lls, *args, **kwargs):
    chains = [mle_ct_single_chain(*args, **kwargs, trail_weights=ll) for ll in lls]
    return Mixture.from_combination(chains)


def continuous_em_learn(n, L, trails_ct, max_iter=100, init=None):
    mixture = Mixture.random(n, L) if init is None else init
    time_lls = 0
    time_mle = 0

    for _ in range(max_iter):
        t = time.time()
        log_lls = likelihood(mixture, trails_ct)
        time_lls += time.time() - t
        lls = np.exp(log_lls - np.max(log_lls, axis=0))
        lls /= np.sum(lls, axis=0)[None, :]
        t = time.time()
        mixture = mle_ct_prior(lls, n, trails_ct)
        time_mle += time.time() - t

    # print("Time spent on likelihood: ", time_lls)
    # print("Time spent on MLE: ", time_mle)
    return mixture

