from collections import defaultdict
# from ctlearn import *
import scipy
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import dtlearn as dt



def sample_trails(S, Ts, n_samples, t_len, return_chains=False):
    L, n = S.shape
    trails = np.zeros((n_samples, t_len), dtype=int)
    flatS = S.flatten()
    starting = np.random.choice(range(len(flatS)), size=n_samples, p=flatS)
    trails[:,0], ls = np.divmod(starting, L)

    rnd = np.random.random_sample((n_samples, t_len-1))
    cum = np.cumsum(Ts, axis=2)
    for i in range(1, t_len):
        intvals = cum[ls, trails[:,i-1]]
        trails[:,i] = np.sum(rnd[:,i-1][:,None] > intvals, axis=1)

    if return_chains: return trails, ls
    else: return trails


class Mixture:
    """A mixture of continuous-time Markov chains."""
    def __init__(self, S, Ks):
        """S is a L x n matrix of starting probabilities. Ks is a L x n x n rate matrix."""
        self.S = np.array(S)
        self.Ks = np.array(Ks)
        self.L, self.n = self.S.shape
        assert(self.S.shape == (self.L, self.n))
        assert(self.Ks.shape == (self.L, self.n, self.n))

    def from_combination(mixtures):
        S, Ks = zip(*[(mixture.S, mixture.Ks) for mixture in mixtures])
        mixture = Mixture(np.vstack(S), np.vstack(Ks))
        mixture.normalize()
        return mixture

    def random(n, L, jtime=None):
        S = np.random.random((L, n))
        S /= np.sum(S)
        Ks = 1 * np.random.random((L, n, n))
        Ks[:, range(n), range(n)] = 0
        Ks[:, range(n), range(n)] = -np.sum(Ks, axis=2)
        if jtime is not None: Ks = - jtime * Ks / Ks[:, range(n), range(n)][:, :, None]
        return Mixture(S, Ks)

    def Ts(self, tau):
        return np.array([scipy.linalg.expm(K * tau) for K in self.Ks])

    def toDTMixture(self, tau):
        return dt.Mixture(self.S, self.Ts(tau))

    def sample(self, n_samples=1, t_len=10, tau=1, return_chains=False):
        Ts = self.Ts(tau)
        return sample_trails(self.S, Ts, n_samples=n_samples, t_len=t_len, return_chains=return_chains)
    
    def sample_ct(self, n_samples=1, duration=1, return_chains=False):
        flatS = self.S.flatten()
        starting = np.random.choice(range(len(flatS)), size=n_samples, p=flatS)
        starting_i, ls = np.divmod(starting, self.L)
        trails_ct = [[] for _ in range(n_samples)]
        for trail_ct, i, l in zip(trails_ct, starting_i, ls):
            K = self.Ks[l]
            total_time = 0
            while total_time < duration:
                time = np.random.exponential(-1 / K[i,i])
                total_time += time
                trail_ct.append((i, time))
                p = K[i] / -K[i,i]
                p[i] = 0
                i = np.random.choice(range(self.n), p=p)
        return (trails_ct, ls) if return_chains else trails_ct

    def __str__(self):
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            x = "Mixture(    # starting probabilities:\n "
            x += str(self.S).replace('\n', '\n ')
            x += "\n,           # rate matrices:\n "
            x += str(self.Ks).replace('\n', '\n ')
            x += "\n)"
            return x
    
    def normalize(self):
        self.S = np.clip(self.S, 0, np.inf)
        self.Ks = np.clip(self.Ks, 0, np.inf)
        self.S /= np.sum(self.S)
        self.Ks[:, range(self.n), range(self.n)] = 0
        self.Ks[:, range(self.n), range(self.n)] = -np.sum(self.Ks, axis=2)
    
    def recovery_error(self, other):
        assert(self.L == other.L)
        assert(not np.isnan(self.Ks).any() and not np.isnan(other.Ks).any()), "NaNs in Ks"

        dists = np.zeros((self.L, self.L))
        for i1, i2 in itertools.product(range(self.L), repeat=2):
            K1 = self.Ks[i1]
            K2 = other.Ks[i2]
            def f(t, x, y):
                return np.abs(K1[x,y] * np.exp(t * K1[x,x]) - K2[x,y] * np.exp(t * K2[x,x]))
            def tv(x):
                return np.sum([scipy.integrate.quad(f, 0, np.inf, args=(x, y))
                               for y in range(self.n) if y != x]) / 2
            dists[i1, i2] = np.mean([tv(x) for x in range(self.n)])

            """
            s1 = -np.diag(self.Ks[i1])
            s2 = -np.diag(other.Ks[i2])
            K1 = self.Ks[i1] + s1 * np.eye(self.n)
            K2 = other.Ks[i2] + s2 * np.eye(self.n)
            s1 = s1[:, None, None] # (x, y, t)
            s2 = s2[:, None, None]
            K1 = K1[:, :, None]
            K2 = K2[:, :, None]

            dt = 1 / (s1 + s2)
            t = np.arange(1000)[None, None, :][[0] * self.n] * dt
            dists[i1, i2] = np.sum(np.abs(K1 * np.exp(t * s1) - K2 * np.exp(t * s2)) * dt) / (2 * self.n)
            import pdb; pdb.set_trace()
            """

        row_ind, col_ind = linear_sum_assignment(dists)
        err = np.sum(dists[row_ind, col_ind]) / self.L
        assert(err <= 1.1)
        return err

    def max_mixing_time(self, eps=1e-10):
        def mixing_time(s, K):
            s /= np.sum(s)
            x = scipy.linalg.expm(K * 1000)[0]
            x /= np.sum(x)

            # exponential search
            tlow = 1
            thigh = None
            t = tlow
            while thigh is None or thigh - tlow > 0.01:
                if thigh is None:
                    t *= 2
                else:
                    t = (tlow + thigh) / 2
                y = s @ scipy.linalg.expm(K * t)
                if np.linalg.norm(x - y) > eps:
                    tlow = t
                else:
                    thigh = t
            return thigh

        return max([mixing_time(s, K) for s, K in zip(self.S, self.Ks)])


