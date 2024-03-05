import itertools
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import *
import ctlearn as ct
import dtlearn as dt
import time



label_tau = "discretization rate $\\tau$"
label_L = "$L$"
label_t_len = "trail length $m$"
label_n_samples = "number of trails $r$"
label_kausik = "KTT" # "Kausik et al." # 2023
label_mcgibbon = "MLE" # "MLE (McGibbon and Pande)" # 2015
label_svd = "GKV-ST" # "SVD (Gupta et al.)" # 2016
label_t_time = "transition time"
label_inf = "Infinitesimal"
label_em = "dEM" # "EM"
label_continuous_em = "cEM" # "EM (continuous sample)"
label_recovery_error = "recovery error"



def clustering_error(x, y, use_max=False):
    if use_max:
        n = len(x)
        x = np.argmax(x, axis=0)[None, :] == np.arange(n)[:, None]
        y = np.argmax(y, axis=0)[None, :] == np.arange(n)[:, None]
        # y = y == np.max(y, axis=0)[None, :]
        # x = x == np.max(x, axis=0)[None, :]
    dists = np.mean(np.abs(x.astype(float)[None, :, :] - y.astype(float)[:, None, :]), axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)
    return np.sum(dists[row_ind, col_ind]) / 2

@memoize
def gen_example(n, L, tau, t_len, n_samples, f=None, sim=None, jumble=False, jtime=None, seed=None):
    mixture = ct.Mixture.random(n, L, jtime=jtime)
    if f is not None: # same topology but different holding time
        assert(f > 0)
        for i in range(L):
            mixture.Ks[i] = f**i * mixture.Ks[i]
    if sim is not None:
        mixture.Ks[1] = sim * mixture.Ks[0] + (1 - sim) * mixture.Ks[1]
    if jumble: # same holding time but different topology
        K = mixture.Ks[0]
        for l in range(1, L):
            for i in range(n):
                permutation = list(np.random.permutation([j for j in range(n) if i != j]))
                permutation.insert(i, i)
                mixture.Ks[l, i] = K[i, permutation]
    trails, chains = mixture.sample(n_samples=n_samples, t_len=t_len, tau=tau, return_chains=True)
    groundtruth = chains[None, :] == np.arange(L)[:, None]
    return mixture, trails, groundtruth


@memoize
def test_methods(n, L, tau, t_len, n_samples, f=None, jumble=None, seed=None):
    mixture, trails, groundtruth = gen_example(n, L, tau, t_len, n_samples, f=f, seed=seed)

    kausik_start_time = time.time()
    kausik_mixture_ct, labels, kausik_mle_time = ct.kausik_learn(n, L, trails, tau, return_labels=True, return_time=True)
    kausik_time = time.time() - kausik_start_time
    kausik_lls = labels[None, :] == np.arange(L)[:, None]
    kausik_mixture_dt = dt.mle(n, trails, kausik_lls)

    em_start_time = time.time()
    # KEVIN: MODIFIED THIS, NOW EVERY DT EM ALGORITHM IS WARM-STARTED BY US
    em_mixture_dt = dt.em_learn(n, L, trails, init=kausik_mixture_dt)
    em_lls = dt.likelihood(em_mixture_dt, trails)
    em_mle_start_time = time.time()
    
    
    em_mixture_ct = ct.mle_prior(em_lls, n, trails, tau=tau)
    em_mle_time = time.time() - em_mle_start_time
    em_time = time.time() - em_start_time
    

    svd_mixture_dt = None
    svd_mixture_ct = None
    svd_lls = None
    svd_time = None
    svd_mle_time = None
    if True: # t_len == 3:
        sample = dt.Distribution.from_trails(n, trails)
        svd_start_time = time.time()
        svd_mixture_dt = dt.svd_learn(sample, n, L)
        svd_lls = dt.likelihood(svd_mixture_dt, trails)
        svd_mle_start_time = time.time()
        svd_mixture_ct = ct.mle_prior(svd_lls, n, trails, tau=tau)
        svd_mle_time = time.time() - svd_mle_start_time
        svd_time = time.time() - svd_start_time

    return {
        'mixture': mixture,
        'trails': trails,
        'groundtruth': groundtruth,

        'kausik_mixture_ct': kausik_mixture_ct,
        'kausik_mixture_dt': kausik_mixture_dt,
        'kausik_lls': kausik_lls,
        'kausik_recovery_error': mixture.recovery_error(kausik_mixture_ct),
        'kausik_clustering_error': clustering_error(groundtruth, kausik_lls),

        'em_mixture_ct': em_mixture_ct,
        'em_mixture_dt': em_mixture_dt,
        'em_lls': em_lls,
        'em_recovery_error': mixture.recovery_error(em_mixture_ct),
        'em_clustering_error': clustering_error(groundtruth, em_lls),

        'svd_mixture_ct': svd_mixture_ct,
        'svd_mixture_dt': svd_mixture_dt,
        'svd_lls': svd_lls,
        'svd_recovery_error': None if svd_mixture_ct is None else mixture.recovery_error(svd_mixture_ct),
        'svd_clustering_error': None if svd_lls is None else clustering_error(groundtruth, svd_lls),

        'kausik_time': kausik_time,
        'kausik_mle_time': kausik_mle_time,
        'em_time': em_time,
        'em_mle_time': em_mle_time,
        'svd_time': svd_time,
        'svd_mle_time': svd_mle_time,
    }

@memoize
def test_clustering_methods(n, L, tau, t_len, n_samples, f=None, seed=None):
    x = test_methods(n, L, tau, t_len, n_samples, f=f, seed=seed)
    return {
        'clustering_error_kausik': x['kausik_clustering_error'],
        'clustering_error_em': x['em_clustering_error'],
        'clustering_error_svd': x['svd_clustering_error'],
    }


@memoize
def test_baseline(n, L, tau, t_len, n_samples, f=None, jumble=None, seed=None, init=None):
    mixture, trails, groundtruth = gen_example(n, L, tau, 1 + int(t_len), n_samples, f=f, jumble=jumble, seed=seed)
    duration = t_len * tau
    trails_ct = mixture.sample_ct(n_samples, duration)

    continuous_em_start_time = time.time()
    continuous_em_mixture = ct.continuous_em_learn(n, L, trails_ct, init=init)
    continuous_em_time = time.time() - continuous_em_start_time
    continuous_em_lls = ct.likelihood(continuous_em_mixture, trails_ct)

    return {
        'mixture': mixture,
        'trails': trails,
        'groundtruth': groundtruth,

        'continuous_em_mixture': continuous_em_mixture,
        'continuous_em_lls': continuous_em_lls,
        'continuous_em_recovery_error': mixture.recovery_error(continuous_em_mixture),
        'continuous_em_clustering_error': clustering_error(groundtruth, continuous_em_lls),
        'continuous_em_time': continuous_em_time,
    }


def test_methods_with_baseline(*args, **kwargs):
    methods = test_methods(*args, **kwargs)
    # KEVIN: INITIALIZING CT EM WITH OUR METHOD, QUICK AND DIRTY WAY TO DO IT
    baseline = test_baseline(*args, **dict(kwargs, init=methods['kausik_mixture_ct']))
    return {
        **baseline,
        **methods,
    }


@genpath
def test_clustering_methods_plot(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_clustering_methods(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("t_len")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.clustering_error_em, label=label_em, zorder=1, **next_config())
        plt.fill_between(mean.index, mean.clustering_error_em - std.clustering_error_em, mean.clustering_error_em + std.clustering_error_em, alpha=0.2)
        plt.plot(mean.index, mean.clustering_error_kausik, label=label_kausik, zorder=1, **next_config())
        plt.fill_between(mean.index, mean.clustering_error_kausik - std.clustering_error_kausik, mean.clustering_error_kausik + std.clustering_error_kausik, alpha=0.2)
        plt.plot(mean.index, mean.clustering_error_svd, label=label_svd, zorder=2, **next_config())
        plt.fill_between(mean.index, mean.clustering_error_svd - std.clustering_error_svd, mean.clustering_error_svd + std.clustering_error_svd, alpha=0.2)

        plt.title(gen_title(setup))
        plt.legend()
        plt.xlabel(label_t_len)
        plt.ylabel("clustering error")
        savefig()


@memoize
def proportional_rates(n, L, tau, t_len, n_samples, f, seed=None, best_of=5):
    x = test_methods(n, L, tau, t_len, n_samples, f=f, seed=seed)
    mixture = x["mixture"]
    groundtruth = x["groundtruth"]
    trails = x["trails"]

    init_mixture = ct.Mixture.random(n, L)
    init_mixture.Ks[1] = f * init_mixture.Ks[0]
    learned_mixture_dt = dt.em_learn(n, L, trails, init=init_mixture.toDTMixture(tau))
    lls = dt.likelihood(learned_mixture_dt, trails)

    init_mixture2 = ct.em_learn_init(n, L, trails, tau=tau)
    learned_mixture_dt2 = dt.em_learn(n, L, trails, init=init_mixture2.toDTMixture(tau))
    lls2 = dt.likelihood(learned_mixture_dt2, trails)

    learned_mixture_ct = ct.mle_prior(lls, n, trails, tau=tau)
    learned_mixture_ct2 = ct.mle_prior(lls2, n, trails, tau=tau)
    perfect_mixture_ct = ct.mle_prior(groundtruth, n, trails, tau=tau)

    # best of 5 for std (bad) initialization
    em5_best_recovery_error = None
    em5_best_clustering_error = None
    for _ in range(best_of):
        em5_mixture_dt = dt.em_learn(n, L, trails, max_iter=20)
        em5_lls = dt.likelihood(em5_mixture_dt, trails)
        em5_mixture_ct = ct.mle_prior(em5_lls, n, trails, tau=tau)
        em5_recovery_error = mixture.recovery_error(em5_mixture_ct)
        em5_clustering_error = clustering_error(groundtruth, em5_lls)
        if em5_best_recovery_error is None or em5_best_recovery_error > em5_recovery_error:
            em5_best_recovery_error = em5_recovery_error
            em5_best_clustering_error = em5_clustering_error

    return {
        "cluster_error": clustering_error(groundtruth, lls),
        "cluster_error2": clustering_error(groundtruth, lls2),
        "recovery_error": mixture.recovery_error(learned_mixture_ct),
        "recovery_error2": mixture.recovery_error(learned_mixture_ct2),
        "recovery_error_mcgibbon": mixture.recovery_error(perfect_mixture_ct),
        "cluster_error_noinit": x['em_clustering_error'],
        "recovery_error_noinit": x['em_recovery_error'],
        "cluster_error_noinit5": em5_best_clustering_error,
        "recovery_error_noinit5": em5_best_recovery_error,
    }


@genpath
def plot_end2end_tau2(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df["t_len"] = (50 / df.tau).astype(int)
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("tau")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label="EM", **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            # plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)

        plt.legend()
        # plt.title(f"$L={setup['L'][0]}$, trail length$={setup['t_len'][0]}$, number of samples$={setup['n_samples'][0]}$")
        plt.title(gen_title(setup))
        plt.xlabel(label_tau)
        plt.ylabel("recovery error")
        savefig()


@genpath
def plot_end2end_tau_jumble(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, jumble=True, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("tau")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label="EM", **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            # plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)


        plt.legend()
        # plt.title(f"$L={setup['L'][0]}$, trail length$={setup['t_len'][0]}$, number of samples$={setup['n_samples'][0]}$")
        plt.title(gen_title(setup))
        plt.xlabel(label_tau)
        plt.ylabel("recovery error")
        savefig()


@genpath
def plot_end2end_tau(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("tau")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label="EM", **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)


        plt.legend()
        # plt.title(f"$L={setup['L'][0]}$, trail length$={setup['t_len'][0]}$, number of samples$={setup['n_samples'][0]}$")
        plt.title(gen_title(setup))
        plt.xlabel(label_tau)
        plt.ylabel("recovery error")
        savefig()


@genpath
def plot_end2end_tlen2(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("t_len")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label=label_em, **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)

        plt.legend()
        plt.title(gen_title(setup))
        plt.xlabel(label_t_len)
        plt.ylabel("recovery error")
        savefig()


@genpath
def plot_end2end_n_samples(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("n_samples")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label=label_em, **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)

        plt.legend(loc="upper right")
        plt.title(gen_title(setup))
        plt.xlabel(label_n_samples)
        plt.ylabel("recovery error")
        savefig()


@genpath
def plot_end2end_tlen(setup, n_observations=1000, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df["n_samples"] = (n_observations / df.t_len).astype(int)
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    # HAD TO FIX THIS -- BAD COLUMNS WERE IN THE DATAFRAME
    x = df[['n', 'L', 'tau', 't_len', 'seed', 'n_samples',
       'continuous_em_recovery_error', 'continuous_em_clustering_error',
       'continuous_em_time', 'kausik_recovery_error', 'kausik_clustering_error', 'em_recovery_error',
       'em_clustering_error',
       'svd_recovery_error', 'svd_clustering_error', 'kausik_time',
       'kausik_mle_time', 'em_time', 'em_mle_time', 'svd_time',
       'svd_mle_time']].groupby("t_len")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label=label_em, **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)

        plt.legend()
        # plt.title(f"$L={setup['L'][0]}$, $\\tau={setup['tau'][0]}$, (number of samples) $\\cdot$ (trail length) = {n_observations}")
        plt.title(gen_title(setup, {"m \\cdot r": n_observations}))
        plt.xlabel(label_t_len)
        plt.ylabel("recovery error")
        savefig()


@genpath
def plot_end2end_L(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df["n_samples"] = df.samples_per_chain * df.L
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    x = df.groupby("L")
    mean = x.mean()
    std = x.std()

    if plt is not None:
        plt.plot(mean.index, mean.em_recovery_error, label="EM", **next_config())
        plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
        plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
        if "svd_recovery_error" in mean.columns:
            plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
            # plt.fill_between(mean.index, mean.svd_recovery_error - std.svd_recovery_error, mean.svd_recovery_error + std.svd_recovery_error, alpha=0.2)
        plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
        plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)

        plt.legend()
        # plt.title(f"$\\tau={setup['tau'][0]}$, number of samples$ = {setup['samples_per_chain'][0]}$ (per chain), trail length={setup['t_len'][0]}")
        plt.title(gen_title(setup))
        plt.xlabel("$L$")
        plt.ylabel("recovery error")
        savefig()


@genpath
def proportional_rates_plot(setup, part=1, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: proportional_rates(row.n, row.L, row.tau, row.t_len, row.n_samples, f=row.f, seed=row.seed, best_of=5))
    # df = df.join(df.astype("object").apply(lambda row: proportional_rates(row.n, row.L, row.tau, row.t_len, row.n_samples, f=row.f, seed=row.seed), axis=1, result_type='expand'))

    x = df.groupby("f")
    mean = x.mean()
    std = x.std()

    # create to axes that share the same x axis
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # ax2 = ax1.twinx()

    recovery_error_config = next_config()
    recovery_error_config["color"] = "tab:cyan"
    recover_error_noinit_config = next_config()
    recover_error_noinit_config["color"] = "tab:brown"
    recover_error_noinit5_config = next_config()
    recovery_error2_config = next_config()
    recovery_error2_config["color"] = "tab:pink"

    if part == 1:
        plt.plot(mean.index, mean["recovery_error"], label="good", **recovery_error_config)
        plt.fill_between(mean.index, mean["recovery_error"] - std["recovery_error"], mean["recovery_error"] + std["recovery_error"], alpha=0.2, color=recovery_error_config["color"])

        plt.plot(mean.index, mean["recovery_error_noinit"], label="random", **recover_error_noinit_config)
        plt.fill_between(mean.index, mean["recovery_error_noinit"] - std["recovery_error_noinit"], mean["recovery_error_noinit"] + std["recovery_error_noinit"], alpha=0.2, color=recover_error_noinit_config["color"])

        # plt.plot(mean.index, mean["recovery_error_noinit5"], label="random (best of 5)", **recover_error_noinit5_config)
        # plt.fill_between(mean.index, mean["recovery_error_noinit5"] - std["recovery_error_noinit5"], mean["recovery_error_noinit5"] + std["recovery_error_noinit5"], alpha=0.2)

        plt.plot(mean.index, mean.recovery_error2, label="learned rates", **recovery_error2_config)
        plt.fill_between(mean.index, mean.recovery_error2 - std.recovery_error2, mean.recovery_error2 + std.recovery_error2, alpha=0.2, color=recovery_error2_config["color"])

        plt.plot(mean.index, mean["recovery_error_mcgibbon"], label="groundtruth", color="black")

        plt.ylabel("recovery error")
        plt.xlabel("factor $f$")
        # plt.set_ylim(-0.1, None)

        plt.legend(loc="upper center", ncol=2, labelspacing=None, columnspacing=1.0)

        plt.title(gen_title(setup))

    elif part == 2:
        plt.plot(mean.index, mean["cluster_error"], label="good", **recovery_error_config)
        plt.fill_between(mean.index, mean["cluster_error"] - std["cluster_error"], mean["cluster_error"] + std["cluster_error"], alpha=0.2, color=recovery_error_config["color"])

        plt.plot(mean.index, mean["cluster_error_noinit"], label="random", **recover_error_noinit_config)
        plt.fill_between(mean.index, mean["cluster_error_noinit"] - std["cluster_error_noinit"], mean["cluster_error_noinit"] + std["cluster_error_noinit"], alpha=0.2, color=recover_error_noinit_config["color"])

        # plt.plot(mean.index, mean["cluster_error_noinit5"], label="random (best of 5)", **recover_error_noinit5_config)
        # plt.fill_between(mean.index, mean["cluster_error_noinit5"] - std["cluster_error_noinit5"], mean["cluster_error_noinit5"] + std["cluster_error_noinit5"], alpha=0.2)

        plt.plot(mean.index, mean["cluster_error2"], label="learned rates", **recovery_error2_config)
        plt.fill_between(mean.index, mean.cluster_error2 - std.cluster_error2, mean.cluster_error2 + std.cluster_error2, alpha=0.2, color=recovery_error2_config["color"])

        plt.ylabel("clustering error")


        plt.xlabel("factor $f$")

        plt.title(gen_title(setup))

    savefig()


@memoize
def single_chain_tau(n, tau, t_time, n_samples, jtime, seed=None):
    t_len = int(t_time / tau)
    mixture, trails, _ = gen_example(n, 1, tau, t_len, n_samples, jtime=jtime, seed=seed)

    learned_mixture_mle = ct.mle_single_chain(n, trails, tau=tau)
    # learned_mixture_inf = ct.infinitesimal_single_chain(n, trails, tau=tau)

    return {
        "recovery_error_mle": mixture.recovery_error(learned_mixture_mle),
        # "recovery_error_inf": mixture.recovery_error(learned_mixture_inf),
    }

@memoize
def single_chain_tau_const_t_len(n, tau, t_len, n_samples, jtime, seed=None):
    mixture, trails, _ = gen_example(n, 1, tau, t_len, n_samples, jtime=jtime, seed=seed)

    learned_mixture_mle = ct.mle_single_chain(n, trails, tau=tau)
    learned_mixture_inf = ct.infinitesimal_single_chain(n, trails, tau=tau)

    return {
        "recovery_error_mle": mixture.recovery_error(learned_mixture_mle),
        "recovery_error_inf": mixture.recovery_error(learned_mixture_inf),
    }

@genpath
def plot_single_chain_tau_const_t_len(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: single_chain_tau_const_t_len(row.n, row.tau, row.t_len, row.n_samples, 1, seed=row.seed))

    if plt is not None:
        for t_len, grp in df.groupby("t_len"):
            x = grp.groupby("tau")
            mean = x.mean()
            std = x.std()
            config = next_config()

            plt.plot(mean.index, mean.recovery_error_mle, label=f"{label_mcgibbon} ({label_t_len}={t_len})", **config)
            plt.fill_between(mean.index, mean.recovery_error_mle - std.recovery_error_mle, mean.recovery_error_mle + std.recovery_error_mle, alpha=0.2, color=config["color"])

            plt.plot(mean.index, mean.recovery_error_inf, label=f"{label_inf} ({label_t_len}={t_len})", **config, linestyle="dashed")
            plt.fill_between(mean.index, mean.recovery_error_inf - std.recovery_error_inf, mean.recovery_error_inf + std.recovery_error_inf, alpha=0.2, color=config["color"])

        plt.legend()
        plt.axvline(x=1, color="black")
        plt.xlabel(label_tau)
        plt.ylabel("recovery error")
        savefig()

@genpath
def plot_single_chain_tau(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: single_chain_tau(row.n, row.tau, row.t_time, row.n_samples, 1, seed=row.seed))

    if plt is not None:
        for t_time, grp in df.groupby("t_time"):
            x = grp.groupby("tau")
            mean = x.mean()
            std = x.std()

            plt.plot(mean.index, mean.recovery_error_mle, label=f"{label_t_time}={t_time}", **next_config())
            plt.fill_between(mean.index, mean.recovery_error_mle - std.recovery_error_mle, mean.recovery_error_mle + std.recovery_error_mle, alpha=0.2)


        plt.legend()
        plt.axvline(x=1, color="black")
        plt.xlabel(label_tau)
        plt.ylabel("recovery error")
        savefig()


@genpath
def test_single_chain_plot(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: single_chain_tau_const_t_len(row.n, row.tau, row.t_len, row.n_samples, None, seed=row.seed))

    for method, label in [("recovery_error_mle", label_mcgibbon), ("recovery_error_inf", label_inf)]:
        x = df.groupby("tau")[method]
        mean = x.mean()
        std = x.std()
        plt.plot(mean.index, mean, label=label, **next_config())
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.xlabel(label_tau)
    plt.ylabel("recovery error")
    savefig()


@memoize
def get_runtimes(n_values, L, tau, t_len, n_samples, max_time=3000, seeds=[None]):

    def dem():
        em_mixture_dt = dt.em_learn(n, L, trails)
        em_lls = dt.likelihood(em_mixture_dt, trails)
        ct.mle_prior(em_lls, n, trails, tau=tau)

    def ktt():
        ct.kausik_learn(n, L, trails, tau)

    def gkv():
        sample = dt.Distribution.from_trails(n, trails)
        svd_mixture_dt = dt.svd_learn(sample, n, L)
        svd_lls = dt.likelihood(svd_mixture_dt, trails)
        ct.mle_prior(svd_lls, n, trails, tau=tau)

    def cem():
        ct.continuous_em_learn(n, L, trails_ct)

    methods = {"dem": dem, "ktt": ktt, "gkv": gkv, "cem": cem}
    times_n = []

    for n in n_values:
        times = {method_name: [] for method_name, _ in methods.items()}

        for seed in seeds:
            mixture, trails, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
            duration = t_len * tau
            trails_ct = mixture.sample_ct(n_samples, duration)

            for method_name, method in methods.items():
                print(f"> running {method_name}")
                start_time = time.time()
                method()
                runtime = time.time() - start_time
                times[method_name].append(runtime)
                print(f"> finished in {runtime} [ms]")
        
        avg_times = {method_name: np.mean(times) for method_name, times in times.items()}
        methods = {method_name: method for method_name, method in methods.items() if avg_times[method_name] < max_time}

        times_n.append(times)
    
    return times_n


@genpath
def plot_scalability_max_n(setup, savefig=None):
    x = get_runtimes(setup["n"], setup["L"][0], setup["tau"][0], setup["t_len"][0], setup["n_samples"][0], max_time=setup["max_time"][0], seeds=setup["seed"])
    df = pd.DataFrame(x, index=setup["n"])

    mean = df.applymap(np.mean)
    std = df.applymap(np.std)
    x_vals = mean.index

    plt.yscale("log")

    plt.plot(x_vals, mean.dem, label=label_em, **next_config())
    plt.fill_between(x_vals, mean.dem - std.dem, mean.dem + std.dem, alpha=0.2)

    plt.plot(x_vals, mean.ktt, label=label_kausik, **next_config())
    plt.fill_between(x_vals, mean.ktt - std.ktt, mean.ktt + std.ktt, alpha=0.2)

    plt.plot(x_vals, mean.gkv, label=label_svd, **next_config())
    plt.fill_between(x_vals, mean.gkv - std.gkv, mean.gkv + std.gkv, alpha=0.2)

    plt.plot(x_vals, mean.cem, label=label_continuous_em, **next_config(), linestyle="dashed")
    plt.fill_between(x_vals, mean.cem - std.cem, mean.cem + std.cem, alpha=0.2)

    plt.title(gen_title(setup))
    plt.legend()
    plt.xlabel("number of states $n$")
    plt.ylabel("time [seconds]")
    savefig()



def plot_scalability(setup, var, label_var, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed))

    x = df.groupby(var)
    mean = x.mean()
    std = x.std()

    plt.plot(mean.index, mean.em_time, label="EM", **next_config())
    plt.fill_between(mean.index, mean.em_time - std.em_time, mean.em_time + std.em_time, alpha=0.2)
    plt.plot(mean.index, mean.kausik_time, label=label_kausik, **next_config())
    plt.fill_between(mean.index, mean.kausik_time - std.kausik_time, mean.kausik_time + std.kausik_time, alpha=0.2)
    if "svd_time" in mean.columns:
        plt.plot(mean.index, mean.svd_time, label=label_svd, **next_config())
    plt.plot(mean.index, mean.continuous_em_time, label=label_continuous_em, **next_config())
    plt.fill_between(mean.index, mean.continuous_em_time - std.continuous_em_time, mean.continuous_em_time + std.continuous_em_time, alpha=0.2)

    plt.legend()
    plt.xlabel(label_var)
    plt.ylabel("time [s]")
    savefig()

@genpath
def plot_scalability_n(setup, savefig=None):
    plot_scalability(setup, "n", "$n$", savefig)

@genpath
def plot_scalability_L(setup, savefig=None):
    plot_scalability(setup, "L", "$L$", savefig)

@genpath
def plot_scalability_tlen(setup, savefig=None):
    plot_scalability(setup, "t_len", label_t_len, savefig)





@memoize
def get_runtimes_L(n, L_values, tau, t_len, n_samples, max_time=3000, seeds=[None]):

    def dem():
        em_mixture_dt = dt.em_learn(n, L, trails)
        em_lls = dt.likelihood(em_mixture_dt, trails)
        ct.mle_prior(em_lls, n, trails, tau=tau)

    def ktt():
        ct.kausik_learn(n, L, trails, tau)

    def gkv():
        sample = dt.Distribution.from_trails(n, trails)
        svd_mixture_dt = dt.svd_learn(sample, n, L)
        svd_lls = dt.likelihood(svd_mixture_dt, trails)
        ct.mle_prior(svd_lls, n, trails, tau=tau)

    def cem():
        ct.continuous_em_learn(n, L, trails_ct)

    methods = {"dem": dem, "ktt": ktt, "gkv": gkv, "cem": cem}
    times_n = []

    for L in L_values:
        times = {method_name: [] for method_name, _ in methods.items()}

        for seed in seeds:
            mixture, trails, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
            duration = t_len * tau
            trails_ct = mixture.sample_ct(n_samples, duration)

            for method_name, method in methods.items():
                print(f"> running {method_name}")
                start_time = time.time()
                method()
                runtime = time.time() - start_time
                times[method_name].append(runtime)
                print(f"> finished in {runtime} [ms]")

        avg_times = {method_name: np.mean(times) for method_name, times in times.items()}
        methods = {method_name: method for method_name, method in methods.items() if avg_times[method_name] < max_time}

        times_n.append(times)

    return times_n


@genpath
def plot_scalability_max_L(setup, savefig=None):
    x = get_runtimes_L(setup["n"][0], setup["L"], setup["tau"][0], setup["t_len"][0], setup["n_samples"][0], max_time=setup["max_time"][0], seeds=setup["seed"])
    df = pd.DataFrame(x, index=setup["L"])

    mean = df.applymap(np.mean)
    std = df.applymap(np.std)
    x_vals = mean.index

    plt.yscale("log")

    plt.plot(x_vals, mean.dem, label=label_em, **next_config())
    plt.fill_between(x_vals, mean.dem - std.dem, mean.dem + std.dem, alpha=0.2)

    plt.plot(x_vals, mean.ktt, label=label_kausik, **next_config())
    plt.fill_between(x_vals, mean.ktt - std.ktt, mean.ktt + std.ktt, alpha=0.2)

    plt.plot(x_vals, mean.gkv, label=label_svd, **next_config())
    plt.fill_between(x_vals, mean.gkv - std.gkv, mean.gkv + std.gkv, alpha=0.2)

    plt.plot(x_vals, mean.cem, label=label_continuous_em, **next_config(), linestyle="dashed")
    plt.fill_between(x_vals, mean.cem - std.cem, mean.cem + std.cem, alpha=0.2)

    plt.title(gen_title(setup))
    # plt.legend()
    plt.xlabel("number of chains $L$")
    plt.ylabel("time [seconds]")
    savefig()





@memoize
def get_runtimes_t_len(n, L, tau, t_len_values, n_samples, max_time=3000, seeds=[None]):

    def dem():
        em_mixture_dt = dt.em_learn(n, L, trails)
        em_lls = dt.likelihood(em_mixture_dt, trails)
        ct.mle_prior(em_lls, n, trails, tau=tau)

    def ktt():
        ct.kausik_learn(n, L, trails, tau)

    def gkv():
        sample = dt.Distribution.from_trails(n, trails)
        svd_mixture_dt = dt.svd_learn(sample, n, L)
        svd_lls = dt.likelihood(svd_mixture_dt, trails)
        ct.mle_prior(svd_lls, n, trails, tau=tau)

    def cem():
        ct.continuous_em_learn(n, L, trails_ct)

    methods = {"dem": dem, "ktt": ktt, "gkv": gkv, "cem": cem}
    times_n = []

    for t_len in t_len_values:
        times = {method_name: [] for method_name, _ in methods.items()}

        for seed in seeds:
            mixture, trails, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
            duration = t_len * tau
            trails_ct = mixture.sample_ct(n_samples, duration)

            for method_name, method in methods.items():
                print(f"> running {method_name}")
                start_time = time.time()
                method()
                runtime = time.time() - start_time
                times[method_name].append(runtime)
                print(f"> finished in {runtime} [ms]")

        avg_times = {method_name: np.mean(times) for method_name, times in times.items()}
        methods = {method_name: method for method_name, method in methods.items() if avg_times[method_name] < max_time}

        times_n.append(times)

    return times_n


@genpath
def plot_scalability_max_t_len(setup, savefig=None):
    x = get_runtimes_t_len(setup["n"][0], setup["L"][0], setup["tau"][0], setup["t_len"], setup["n_samples"][0], max_time=setup["max_time"][0], seeds=setup["seed"])
    df = pd.DataFrame(x, index=setup["t_len"])

    mean = df.applymap(np.mean)
    std = df.applymap(np.std)
    x_vals = mean.index

    plt.yscale("log")
    # plt.ylim(bottom=0.1)

    plt.plot(x_vals, mean.dem, label=label_em, **next_config())
    plt.fill_between(x_vals, mean.dem - std.dem, mean.dem + std.dem, alpha=0.2)

    plt.plot(x_vals, mean.ktt, label=label_kausik, **next_config())
    plt.fill_between(x_vals, mean.ktt - std.ktt, mean.ktt + std.ktt, alpha=0.2)

    plt.plot(x_vals, mean.gkv, label=label_svd, **next_config())
    plt.fill_between(x_vals, mean.gkv - std.gkv, mean.gkv + std.gkv, alpha=0.2)

    plt.plot(x_vals, mean.cem, label=label_continuous_em, **next_config(), linestyle="dashed")
    plt.fill_between(x_vals, mean.cem - std.cem, mean.cem + std.cem, alpha=0.2)

    plt.title(gen_title(setup))
    plt.legend()
    plt.xlabel(label_t_len)
    # plt.ylabel("time [seconds]")
    savefig()




@memoize
def sample_complexity_kausik(n, L, tau, t_len, n_samples, seed):
    mixture, trails, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
    kausik_mixture_ct, _, _ = ct.kausik_learn(n, L, trails, tau, return_labels=True, return_time=True)
    return mixture.recovery_error(kausik_mixture_ct)

@memoize
def sample_complexity_em(n, L, tau, t_len, n_samples, seed):
    mixture, trails, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
    em_mixture_dt = dt.em_learn(n, L, trails)
    em_lls = dt.likelihood(em_mixture_dt, trails)
    em_mixture_ct = ct.mle_prior(em_lls, n, trails, tau=tau)
    return mixture.recovery_error(em_mixture_ct)

@memoize
def sample_complexity_continuous_em(n, L, tau, t_len, n_samples, seed):
    mixture, _, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
    duration = t_len * tau
    trails_ct = mixture.sample_ct(n_samples, duration)
    continuous_em_mixture = ct.continuous_em_learn(n, L, trails_ct)
    return mixture.recovery_error(continuous_em_mixture)

@memoize
def sample_complexity_svd(n, L, tau, t_len, n_samples, seed):
    mixture, trails, _ = gen_example(n, L, tau, t_len, n_samples, seed=seed)
    sample = dt.Distribution.from_trails(n, trails)
    svd_mixture_dt = dt.svd_learn(sample, n, L)
    svd_lls = dt.likelihood(svd_mixture_dt, trails)
    svd_mixture_ct = ct.mle_prior(svd_lls, n, trails, tau=tau)
    return mixture.recovery_error(svd_mixture_ct)

def sample_complexity(learner, threshold, n, L, tau, t_len, seed):
    learner_map = {
        "kausik": sample_complexity_kausik,
        "em": sample_complexity_em,
        "svd": sample_complexity_svd,
        "continuous_em": sample_complexity_continuous_em,
    }
    n_fails = 0
    n_samples = 300
    final_n_samples = None
    factor = 2.0

    n_iter = 1
    while n_fails < 5 and n_samples >= 2:
        # print(f"Testing {learner} with {n_samples} samples:")
        recovery_error = learner_map[learner](n, L, tau, t_len, n_samples, 100*seed + n_iter)
        # import pdb; pdb.set_trace()
        if recovery_error < threshold: # success
            final_n_samples = n_samples if final_n_samples == None else \
                (0.5 * final_n_samples + 0.5 * n_samples)
            n_samples /= factor
            # print(f"   success (recovery_error={recovery_error}))")
        else: # failure
            n_samples *= factor
            n_fails += 1
            # print(f"   failure (recovery_error={recovery_error}))")
        n_samples = round(n_samples)
        n_iter += 1

    return {
        "n_samples": final_n_samples
    }


@genpath
def plot_sample_complexity(setup, groupby, savefig=None):
    learner_map = {
        "kausik": label_kausik,
        "em": label_em,
        "svd": label_svd,
        "continuous_em": label_continuous_em,
    }
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: sample_complexity(
        row.learner, row.threshold, row.n, row.L, row.tau,
        row.t_len, seed=row.seed))

    plt.ylim(bottom=4, top=3000)
    plt.yscale("log")

    order = ["em", "kausik", "svd", "continuous_em"]
    while len(order) > 0:
        for learner, grp in df.groupby("learner"):
            if len(order) > 0 and order[0] == learner:
                order = order[1:]
                grp.drop("learner", axis=1, inplace=True)
                x = grp.groupby(groupby)
                mean = x.mean()
                std = x.std()

                plt.plot(mean.index, mean.n_samples, label=learner_map[learner], **next_config())
                plt.fill_between(mean.index, mean.n_samples - std.n_samples, mean.n_samples + std.n_samples, alpha=0.2)

    plt.legend()
    plt.xlabel({"t_len": label_t_len, "tau": label_tau}[groupby])
    plt.ylabel("number of samples")
    plt.title(gen_title(setup, extra_space=groupby == "t_len"))
    # plt.title(f"Sample complexity $>{setup['threshold'][0]}$ ($\\tau={setup['tau'][0]}$, trail-length$={setup['t_len'][0]}$)")
    savefig()


def test_kausik(n, L, n_samples, seed):
    tau = 0.1
    t_len = 1000
    mixture, trails, groundtruth = gen_example(n, L, tau, t_len, n_samples, seed=seed)

    tau1 = tau
    t_len1 = t_len // 10
    trails1 = trails[:, :t_len1]
    groundtruth1 = groundtruth[:t_len1]

    t_len2 = t_len1
    tau2 = tau * (t_len // t_len2)
    trails2 = trails[:, ::t_len // t_len2]
    groundtruth2 = groundtruth[::t_len // t_len2]

    print(f"max_mixing_time={mixture.max_mixing_time()}")

    mixture_ct, labels = ct.kausik_learn(n, L, trails, tau=tau, return_labels=True)
    lls = labels[None, :] == np.arange(L)[:, None]
    clust_err = clustering_error(groundtruth, lls)
    recov_err = mixture.recovery_error(mixture_ct)

    mixture_ct1, labels1 = ct.kausik_learn(n, L, trails1, tau=tau1, return_labels=True)
    lls1 = labels1[None, :] == np.arange(L)[:, None]
    clust_err1 = clustering_error(groundtruth1, lls1)
    recov_err1 = mixture.recovery_error(mixture_ct1)

    mixture_ct2, labels2 = ct.kausik_learn(n, L, trails2, tau=tau2, return_labels=True)
    lls2 = labels2[None, :] == np.arange(L)[:, None]
    clust_err2 = clustering_error(groundtruth2, lls2)
    recov_err2 = mixture.recovery_error(mixture_ct2)

    print(f"tau={tau}, t_len={t_len}: clustering_error={clust_err}, recovery_error={recov_err}")
    print(f"tau={tau1}, t_len={t_len1}: clustering_error={clust_err1}, recovery_error={recov_err1}")
    print(f"tau={tau2}, t_len={t_len2}: clustering_error={clust_err2}, recovery_error={recov_err2}")


def nba_predict_outcome(trail_ct_prefix, mixture):
    player, _ = trail_ct_prefix[-1]
    log_ll = ct.likelihood(mixture, [trail_ct_prefix])
    ll = np.exp(log_ll - np.max(log_ll))
    ll /= np.sum(ll)
    probs_per_chain = ll * (np.eye(mixture.n)[player] @ mixture.Ts(10000))[:, [0,1]]
    probs = np.sum(probs_per_chain, axis=0)
    # probs /= np.sum(probs)
    return probs


def nba_prefix_trail(trail_ct, t):
    trail_ct_prefix = []
    total_time = 0
    for player, time in trail_ct:
        if total_time + time < t:
            trail_ct_prefix.append((player, time))
        else:
            trail_ct_prefix.append((player, t - total_time))
            break
        total_time += time
    return trail_ct_prefix


def nba_prediction_error(df, mixture):
    # error = []
    # confidence = []
    score_probs = []
    scored = []
    i = 0
    for _, row in df.iterrows():
        trail_ct = row.trail_ct
        time = sum([t for _, t in trail_ct])
        # print(f"(points={row.ptsScored}, time={time}) {trail_ct}")
        for t in np.linspace(1, time, 20):
            trail_ct_prefix = nba_prefix_trail(trail_ct, t)
            probs = nba_predict_outcome(trail_ct_prefix, mixture)
            miss_prob, score_prob = probs # / np.sum(probs)

            score_probs.append(score_prob)
            scored.append(row.ptsScored > 0)

            # error.append(miss_prob if row.ptsScored > 0 else score_prob)
            # confidence.append(score_prob + miss_prob)
            # print(f"{t:.2f}s: score-prob={100*outcome:.2f}%")
        
        i += 1
        if i > 100: break
        # import pdb; pdb.set_trace()

    # print(f"{i}: error={np.mean(error)} confidence={np.mean(confidence)}")

    score_probs = np.array(score_probs)
    scored = np.array(scored)
    x = [np.mean(np.abs(scored.astype(float) - (score_probs > c))) for c in np.linspace(0, 1, 1000)]
    return np.min(x)


@memoize
def nba_correlation(L, team, season, tau=0.1, max_trail_time=20, min_trail_time=10, test_size=0.25, use_test=True, seed=0):
    # is there correlation between trail likelihood and points scored? 

    import NBA.learn as nba_learn

    data = lambda split: nba_learn.NBADataset(split, team, season, tau=tau, max_trail_time=max_trail_time,
                                              min_trail_time=min_trail_time, test_size=test_size, use_position=True, seed=seed)
    train_iter, test_iter = data(split="train"), data(split="test")

    trails_dt = train_iter.get_trails_dt()
    n = len(train_iter.state_dict)
    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=100)

    iter = test_iter if use_test else train_iter
    trails_ct = iter.get_trails_ct()
    lls_ct = np.exp(np.sum(ct.likelihood(mixture_ct, trails_ct), axis=0))
    labels = iter.get_labels()
    return {
            "miss_ll": lls_ct[labels == 0],
            "score_ll": lls_ct[labels == 1],
    }


@genpath
def plot_nba_correlation(setup, savefig=None):
    setup["team"] = ["GSW", "LAL", "BOS", "MIA", "LAC", "HOU"]
    setup["season"] = [2022]
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: nba_correlation(row.L, row.team, row.season, tau=row.tau, max_trail_time=row.max_trail_time, min_trail_time=row.min_trail_time, use_test=True, seed=row.seed),
                  axis=1, result_type='expand'))

    plt.yscale("log")
    x_axis = np.array([0, 1.5])
    groups = df.groupby("team")
    for i, (team, grp) in enumerate(groups):
        miss_ll = np.concatenate(grp.miss_ll.to_numpy())
        score_ll = np.concatenate(grp.score_ll.to_numpy())
        offset = (i+0.5) / len(groups) - 0.5
        mean = np.array([np.mean(miss_ll), np.mean(score_ll)])
        std = np.array([np.std(miss_ll), np.std(score_ll)])
        plt.bar(x_axis - offset, mean, 1 / len(groups) - 0.05, label=team)
        plt.errorbar(x_axis - offset, mean, std, fmt="o", color="r")
        print(f"{team}: miss_ll={np.mean(miss_ll)} +/-{np.std(miss_ll)}, score_ll={np.mean(score_ll)} +/-{np.std(score_ll)}")

    plt.legend(loc="upper center")
    plt.title(gen_title(setup))
    plt.xticks(x_axis, ["Miss", "Score"])
    plt.ylabel("Likelihood")
    savefig()
 

def nba_print_mixture(mixture, state_dict, trails_ct=None):
    def state_name(i):
        target_len = 12
        state = state_dict[i]
        if isinstance(state, tuple):
            s = (state[1] + "-" + state[0]).replace(' ', '')
        else:
            s = state
        s = s[:target_len-1] + "." if len(s) > target_len else s
        return s + (" " * (target_len - len(s)))

    def print_chain(S, K, T, l):
        starting_prob = np.sum(S)
        miss_prob, score_prob = (S @ T)[[0, 1]] / starting_prob
        print(f"miss={100*miss_prob:.2f}%, score={100*score_prob:.2f}% ({100*starting_prob:.2f}%)")
        if trails_ct is not None:
            log_ll = ct.likelihood(mixture, trails_ct)
            ll = np.exp(log_ll - np.max(log_ll, axis=0))
            ll /= np.sum(ll, axis=0)[None, :]
            print("total likelihood:", np.sum(ll[l]))

        names = [state_name(i) for i in range(len(S))]
        print(" " * 14 + " ".join(names))
        with np.printoptions(precision=10, suppress=True, linewidth=np.inf):
            print(" " * 12, S)
        with np.printoptions(precision=9, suppress=True, linewidth=np.inf):
            print("\n".join([f"{name}{line}" for name, line in zip(names, str(K).split('\n'))]))

    mixture_stationary = mixture.Ts(10000)
    for l in range(mixture.L):
        print(f"\nChain {l}:")
        print_chain(mixture.S[l], mixture.Ks[l], mixture_stationary[l], l)

    real_players = ["PG", "SG", "PF", "SF", "C"]
    baskets = ["miss", "score"]
    for s, K in zip(mixture.S, mixture.Ks):
        print("\n\n")
        for i1, p1 in state_dict.items():
            if p1 not in real_players: continue
            print("    \\node[label={[label distance=6pt]" + ("below" if p1 in ["PF", "SF"] else "above") + ":{" + f"{-10 * K[i1,i1]:.1f}" + "s}}] at (" + p1 + ") {};")
            if s[i1] == max(s):
                print("    \\node[start] at (" + p1 + ") {};")
            for i2, p2 in state_dict.items():
                if p2 in real_players or p2 in baskets:
                    x = - K[i1, i2] / K[i1, i1]
                    if (p2 in real_players and x < 0.2) or (p2 in baskets and x < 0.15): continue
                    color = ("," + {"score": "green", "miss": "red"}[p2]) if p2 in baskets else ""
                    print("    \\draw[pass,opacity=" + str(x) + ",line width=" + str(x * 10) + "pt" + color + "] (" + p1 + ") to (" + p2 + ");")





@memoize
def real_world_nba_nn(team, season, tau=0.1, max_trail_time=20, min_trail_time=10, test_size=0.25, seed=0):
    import NBA.learn as nba_learn
    data = lambda split: nba_learn.NBADataset(split, team, season, tau=tau, max_trail_time=max_trail_time, min_trail_time=min_trail_time, test_size=test_size, seed=seed)
    return nba_learn.text_classification_learn(data)


@memoize
def real_world_nba(L, team, season, tau=0.1, max_trail_time=20, min_trail_time=10, test_size=0.25, seed=0):
    import NBA.learn as nba_learn

    # tau = 0.1
    data = lambda split: nba_learn.NBADataset(split, team, season, tau=tau, max_trail_time=max_trail_time, min_trail_time=min_trail_time, test_size=test_size, seed=seed)
    train_iter, test_iter = data(split="train"), data(split="test")

    def accuracy(mixture):
        score_probs = []
        for trail_ct, label in zip(train_iter.get_trails_ct(), train_iter.get_labels()):
            probs = nba_predict_outcome(trail_ct, mixture)
            miss_prob, score_prob = probs
            score_probs.append(score_prob)
        thresh = np.linspace(0, 1, 1000)
        accu = [np.mean(np.abs(train_iter.get_labels() == (score_probs > c))) for c in np.linspace(0, 1, 1000)]
        t_ix = np.argmax(accu)
        train_accuracy = accu[t_ix]
        t = thresh[t_ix]
        print("threshold:", t)
        print("train accuracy:", np.max([np.mean(np.abs(train_iter.get_labels() == (score_probs > c))) for c in np.linspace(0, 1, 1000)]))

        correct = []
        for trail_ct, label in zip(test_iter.get_trails_ct(), test_iter.get_labels()):
            probs = nba_predict_outcome(trail_ct, mixture)
            miss_prob, score_prob = probs
            correct.append((score_prob > t) == label)

        print("test accuracy:", np.mean(correct))
        return train_accuracy, np.mean(correct)

    # text_classification_accuracy = nba_learn.text_classification_learn(data)
    text_classification_train_accuracy, text_classification_test_accuracy = \
        real_world_nba_nn(team, season, tau=tau, max_trail_time=max_trail_time, min_trail_time=min_trail_time, test_size=test_size, seed=seed)

    trails_dt = train_iter.get_trails_dt()
    n = len(train_iter.state_dict)
    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=100)

    mixture_ct_kausik = ct.kausik_learn(n, L, trails_dt[:100], tau, mle_max_iter=100)

    trails_ct_ = [trail_ct + [(label, 1)]
                  for trail_ct, label in zip(train_iter.get_trails_ct(), train_iter.get_labels())]
    mixture_ct_ = ct.continuous_em_learn(n, L, trails_ct_, max_iter=100)

    em_train_accuracy, em_test_accuracy = accuracy(mixture_ct)
    continuous_em_train_accuracy, continuous_em_test_accuracy = \
        accuracy(mixture_ct_)
    kausik_train_accuracy, kausik_test_accuracy = accuracy(mixture_ct_kausik)

    # import pdb; pdb.set_trace()

    return {
        "text_classification_train_accuracy": text_classification_train_accuracy,
        "em_train_accuracy": em_train_accuracy,
        "kausik_train_accuracy": kausik_train_accuracy,
        "continuous_em_train_accuracy": continuous_em_train_accuracy,

        "text_classification_test_accuracy": text_classification_test_accuracy,
        "emtest__accuracy": em_test_accuracy,
        "kausik_test_accuracy": kausik_test_accuracy,
        "continuous_em_test_accuracy": continuous_em_test_accuracy,
    }


def nba_eval(L, team, season, tau=0.1, max_trail_time=20, min_trail_time=10, test_size=0.25, use_position=True, seed=0):
    import NBA.learn as nba_learn

    data = lambda split: nba_learn.NBADataset(split, team, season, tau=tau, max_trail_time=max_trail_time,
                                              min_trail_time=min_trail_time, test_size=test_size, use_position=use_position, seed=seed)
    train_iter, _ = data(split="train"), data(split="test")

    trails_dt = train_iter.get_trails_dt()
    n = len(train_iter.state_dict)
    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=100)

    nba_print_mixture(mixture_ct, train_iter.state_dict, train_iter.get_trails_ct())


@memoize
def nba_live(L, team, season, tau=0.1, max_trail_time=20, min_trail_time=10, test_size=0.25, seed=0):
    import NBA.learn as nba_learn

    data = lambda split: nba_learn.NBADataset(split, team, season, tau=tau, max_trail_time=max_trail_time,
                                              min_trail_time=min_trail_time, test_size=test_size, use_position=True, seed=seed)
    train_iter, test_iter = data(split="train"), data(split="test")

    def trail_tte(trail, t): # time to end
        total_time = sum([t for _, t in trail])
        if t > total_time: return None
        return nba_prefix_trail(trail, total_time - t)

    def accuracy(mixture):
        correct_tte = {t: [] for t in range(1, 25)}

        for trail_ct, label in zip(test_iter.get_trails_ct(), test_iter.get_labels()):
            for t, correct in correct_tte.items():
                trail_ct_prefix = trail_tte(trail_ct, t)
                if trail_ct_prefix is not None:
                    probs = nba_predict_outcome(trail_ct_prefix, mixture)
                    miss_prob, score_prob = probs
                    correct.append(score_prob if label == 1 else miss_prob)

        mean_tte = {t: np.mean(correct) for t, correct in correct_tte.items()}
        return list(mean_tte.keys()), list(mean_tte.values())

    trails_dt = train_iter.get_trails_dt()
    n = len(train_iter.state_dict)
    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=100)

    tte, accs = accuracy(mixture_ct)
    return {"accs": accs}


@genpath
def plot_nba_live(setup, savefig=None):
    setup["team"] = ["GSW", "LAL", "BOS", "MIA", "LAC", "HOU"]
    setup["season"] = [2022]
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: nba_live(row.L, row.team, row.season, tau=row.tau, max_trail_time=row.max_trail_time, min_trail_time=row.min_trail_time, seed=row.seed),
                  axis=1, result_type='expand'))

    x = np.array(df.accs.tolist())
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    plt.plot(range(len(mean)), mean, label=label_em, **next_config())
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.title(gen_title(setup))
    plt.xlabel(label_L)
    plt.ylabel("Accuracy")
    savefig()


# @memoize
def real_world_nba_test(L, team, season, verbose=True):
    import NBA.extract as nba

    df, state_dict = nba.load_trails(team, season, tau=0.1, model_score=True, verbose=verbose)
    trails_dt = np.array(df.trail_dt.tolist())
    n = len(state_dict)

    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=10000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    tau = 0.1
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=1000)

    trails_ct_ = [row.trail_ct + [(1 if row.ptsScored > 0 else 0, 1)] for _, row in df.iterrows()]
    mixture_ct_ = ct.continuous_em_learn(n, L, trails_ct_, max_iter=100)

    trails_ct = df.trail_ct.tolist()
    lls_ct = ct.likelihood(mixture_ct, trails_ct)
    lls_ct_ = ct.likelihood(mixture_ct_, trails_ct)
    ls_ct = np.exp(lls_ct)
    ls_ct_ = np.exp(lls_ct_)
    explainability = np.sum(ls_ct, axis=0)
    explainability_ = np.sum(ls_ct_, axis=0)
    ls_ct_sorted = np.sort(ls_ct, axis=0)[::-1, :]
    ls_ct_sorted_ = np.sort(ls_ct_, axis=0)[::-1, :]
    discrimination = ls_ct_sorted[0, :] - ls_ct_sorted[1, :]
    discrimination_ = ls_ct_sorted_[0, :] - ls_ct_sorted_[1, :]

    prediction_error = nba_prediction_error(df, mixture_ct)
    prediction_error_ = nba_prediction_error(df, mixture_ct_)

    if verbose:
        print("CT explainability:\n", pd.Series(explainability).describe().round(5))
        print("CT discrimination:\n", pd.Series(discrimination).describe().round(5))
        nba_print_mixture(mixture_ct, state_dict)
        print("prediction_error", prediction_error)


    return {
        "em_explainability": explainability,
        "em_discrimination": discrimination,
        "continuous_em_explainability": explainability_,
        "continuous_em_discrimination": discrimination_,
        "em_prediction_error": prediction_error,
        "continuous_em_prediction_error": prediction_error_,
    }


@genpath
def plot_real_world(setup, savefig=None):
    setup["team"] = ["GSW", "LAL", "BOS", "MIA", "LAC", "HOU"]
    setup["season"] = [2022]
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: real_world_nba(row.L, row.team, row.season, tau=row.tau, max_trail_time=row.max_trail_time, min_trail_time=row.min_trail_time, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("L")

    x = grp.emtest__accuracy
    mean = x.mean()
    std = x.std()
    em_config = next_config()
    line_solid, = plt.plot(mean.index, mean, color="black", label="test")
    line1, = plt.plot(mean.index, mean, label=label_em, **em_config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    line_dashed, = plt.plot(mean.index, grp.em_train_accuracy.mean(), color="black", linestyle="dashed", label="train")
    plt.plot(mean.index, grp.em_train_accuracy.mean(), **em_config, linestyle="dashed")

    x = grp.kausik_test_accuracy
    mean = x.mean()
    std = x.std()
    kausik_config = next_config()
    line2, = plt.plot(mean.index, mean, label=label_kausik, **kausik_config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    plt.plot(mean.index, grp.kausik_train_accuracy.mean(), **kausik_config, linestyle="dashed")

    next_config()

    x = grp.continuous_em_test_accuracy
    mean = x.mean()
    std = x.std()
    continuous_em_config = next_config()
    line3, = plt.plot(mean.index, mean, label=label_continuous_em, **continuous_em_config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=continuous_em_config["color"])
    plt.plot(mean.index, grp.continuous_em_train_accuracy.mean(), **continuous_em_config, linestyle="dashed")

    x = grp.text_classification_test_accuracy
    mean = x.mean()
    std = x.std()
    text_classification_config = next_config()
    line4, = plt.plot(mean.index, mean, label="Neural Network", **text_classification_config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=text_classification_config["color"])
    plt.plot(mean.index, grp.text_classification_train_accuracy.mean(), **text_classification_config, linestyle="dashed")

    legend1 = plt.legend(handles=[line1, line2, line3, line4], loc="upper left")
    plt.gca().add_artist(legend1)
    plt.legend(handles=[line_solid, line_dashed], loc="lower right")

    plt.title(gen_title(setup, {"n": 7}))
    plt.xlabel(label_L)
    plt.ylabel("accuracy")
    savefig()




@genpath
def plot_continuous_em(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("t_len")

    x = grp.continuous_em_recovery_error
    mean = x.mean()
    std = x.std()
    plt.plot(mean.index, mean, label=label_continuous_em, zorder=1, **next_config())
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.title(gen_title(setup))
    plt.xlabel("duration")
    plt.ylabel(label_recovery_error)
    savefig()



@genpath
def plot_end2end_n_samples_with_baseline(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_methods_with_baseline(row.n, row.L, row.tau, row.t_len, row.n_samples, seed=row.seed),
                  axis=1, result_type='expand'))


    x = df.groupby("n_samples")
    mean = x.mean()
    std = x.std()

    plt.plot(mean.index, mean.em_recovery_error, label="EM", **next_config())
    plt.fill_between(mean.index, mean.em_recovery_error - std.em_recovery_error, mean.em_recovery_error + std.em_recovery_error, alpha=0.2)

    plt.plot(mean.index, mean.kausik_recovery_error, label=label_kausik, **next_config())
    plt.fill_between(mean.index, mean.kausik_recovery_error - std.kausik_recovery_error, mean.kausik_recovery_error + std.kausik_recovery_error, alpha=0.2)
    if "svd_recovery_error" in mean.columns:
        plt.plot(mean.index, mean.svd_recovery_error, label=label_svd, **next_config())
    plt.plot(mean.index, mean.continuous_em_recovery_error, label=label_continuous_em, **next_config(), linestyle="dashed")
    plt.fill_between(mean.index, mean.continuous_em_recovery_error - std.continuous_em_recovery_error, mean.continuous_em_recovery_error + std.continuous_em_recovery_error, alpha=0.2)

    plt.legend()
    plt.title(gen_title(setup))
    plt.xlabel(label_n_samples)
    plt.ylabel("recovery error")
    savefig()


@memoize
def test_recovery(n, L, tau, t_len, n_samples, max_iter, seed):
    mixture_ct, trails, groundtruth = gen_example(n, L, tau, t_len, n_samples, seed=seed)
    mixture_dt = mixture_ct.toDTMixture(tau)

    cluster_soft = dt.likelihood(mixture_dt, trails)
    cluster_hard = np.arange(L)[:, None] == np.argmax(cluster_soft, axis=0)

    mixture_ct_true = ct.mle_prior(groundtruth, n, trails, tau=tau, max_iter=max_iter)
    mixture_ct_hard = ct.mle_prior(cluster_hard, n, trails, tau=tau, max_iter=max_iter)
    mixture_ct_soft = ct.mle_prior(cluster_soft, n, trails, tau=tau, max_iter=max_iter)

    return {
        "true": mixture_ct.recovery_error(mixture_ct_true),
        "hard": mixture_ct.recovery_error(mixture_ct_hard),
        "soft": mixture_ct.recovery_error(mixture_ct_soft),
        "cluster_err_hard": clustering_error(groundtruth, cluster_hard),
        "cluster_err_soft": clustering_error(groundtruth, cluster_soft),
    }


@genpath
def plot_recovery(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_recovery(row.n, row.L, row.tau, row.t_len, row.n_samples, max_iter=row.max_iter, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("t_len")

    x = grp["true"]
    mean = x.mean()
    std = x.std()
    plt.plot(mean.index, mean, label="Groundtruth", **next_config())
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["hard"]
    mean = x.mean()
    std = x.std()
    plt.plot(mean.index, mean, label="Hard clustering", **next_config())
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["soft"]
    mean = x.mean()
    std = x.std()
    plt.plot(mean.index, mean, label="Soft clustering", **next_config())
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.title(gen_title(setup))
    plt.xlabel(label_t_len)
    plt.ylabel("recovery error")
    savefig()


def test_recovery_single(n, L, tau, t_len, n_samples, max_iter, seed):
    mixture_ct, trails, groundtruth = gen_example(n, 2, tau, t_len, n_samples, seed=seed)
    mixture_dt = mixture_ct.toDTMixture(tau)

    mixture_ct0 = ct.Mixture(mixture_ct.S[[0]], mixture_ct.Ks[[0]])
    mixture_ct0.normalize()

    cluster_soft = dt.likelihood(mixture_dt, trails)

    mixture_ct_true = ct.mle_prior(groundtruth[[0]], n, trails, tau=tau, max_iter=max_iter)
    mixture_ct_soft = ct.mle_prior(cluster_soft[[0]], n, trails, tau=tau, max_iter=max_iter)

    return {
        "true": mixture_ct0.recovery_error(mixture_ct_true),
        "soft": mixture_ct0.recovery_error(mixture_ct_soft),
    }


# x = test_recovery_single(n=10, L=2, tau=0.1, t_len=10, n_samples=1000, max_iter=100, seed=0)
# print("true", x["true"])
# print("soft", x["soft"])


@memoize
def test_lastfm(n, L, tau, t_len, users, seed=None):
    import LastFM.lastfm as lastfm
    from scipy.stats import entropy
    n_rows = None

    ct_train, ct_test = lastfm.load_ct(
        max_time_delta=lastfm.max_time_delta,
        min_trail_time=lastfm.min_trail_time,
        top_n_songs=n,
        n_rows=n_rows)
    trails_ct_train, labels_ct_train = ct_train
    trails_ct_test, labels_ct_test = ct_test

    # from sklearn.model_selection import train_test_split
    # trails_ct_train, trails_ct_test = train_test_split(trails_ct, test_size=0.2, random_state=seed)

    # labels_ct = [label for label, _ in trails_ct]
    # trails_ct = [[(x, t) for (t, x) in trail] for _, trail in trails_ct]

    dt_train, dt_test = lastfm.load_dt(
        max_time_delta=lastfm.max_time_delta,
        min_trail_time=lastfm.min_trail_time,
        top_n_songs=n,
        n_rows=n_rows,
        tau=tau,
        t_len=t_len)
    trails_dt_train, labels_dt_train = dt_train
    # trails_dt_test, labels_dt_test = dt_test

    if users is not None:
        xs, cs = np.unique(labels_dt_train, return_counts=True)
        ix = np.argpartition(cs, -users)[-users:]
        max_labels = xs[ix]

        ix_dt_train = np.isin(labels_dt_train, max_labels)
        trails_dt_train = trails_dt_train[ix_dt_train]
        labels_dt_train = labels_dt_train[ix_dt_train]

        # ix_dt_test = np.isin(labels_dt_test, max_labels)
        # trails_dt_test = trails_dt_test[ix_dt_test]
        # labels_dt_test = labels_dt_test[ix_dt_test]

        unique_labels = np.unique(max_labels)
        num_unique_labels = len(unique_labels)
        labels_ix = {label: ix for label, ix in zip(unique_labels, range(num_unique_labels))}

        trails_ct_train = [trail for trail, label in zip(trails_ct_train, labels_ct_train) if label in max_labels]
        trails_ct_test = [trail for trail, label in zip(trails_ct_test, labels_ct_test) if label in max_labels]

        labels_ct_train = np.array([labels_ix[label] for label in labels_ct_train if label in max_labels])
        labels_ct_test = np.array([labels_ix[label] for label in labels_ct_test if label in max_labels])

        groundtruth_ct_test = labels_ct_test[None, :] == np.arange(num_unique_labels)[:, None]
        groundtruth_ct_train = labels_ct_train[None, :] == np.arange(num_unique_labels)[:, None]
        
    
    trails_dt_kausik = trails_dt_train[:100]
    mixture_ct_kausik, labels, kausik_mle_time = ct.kausik_learn(n, L, trails_dt_kausik, tau,
                                        mle_max_iter=100, return_labels=True, return_time=True)
    kausik_time = time.time() - kausik_start_time
    kausik_lls = labels[None, :] == np.arange(L)[:, None]
    kausik_mixture_dt = dt.mle(n, trails, kausik_lls)


    # KEVIN: SAME THING HERE, INITIALIZED WITH OUR METHOD
    mixture_dt = dt.em_learn(n, L, trails_dt_train, init=kausik_mixture_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt_train)
    mixture_ct_em = ct.mle_prior(lls_dt, n, trails_dt_train, tau=tau, max_iter=100)


    # trails_dt_svd = trails_dt[:]
    # mixture_ct_svd = ct.svd_learn(n, L, trails_dt_svd, tau, mle_max_iter=100)

    groundtruth_dt_train = labels_dt_train[None, :] == unique_labels[:, None]
    mixture_ct_gt = ct.mle_prior(groundtruth_dt_train, n, trails_dt_train, tau=tau, max_iter=100)

    # KEVIN: WARM-STARTED CT-EM WITH OUR METHOD AS WELL
    mixture_ct_em_continuous = ct.continuous_em_learn(n, L, trails_ct_train, max_iter=100, init=mixture_ct_kausik)

    if users is not None:
        lls_ct_em_test = ct.soft_clustering(mixture_ct_em, trails_ct_test)
        lls_ct_kausik_test = ct.soft_clustering(mixture_ct_kausik, trails_ct_test)
        lls_ct_em_continuous_test = ct.soft_clustering(mixture_ct_em_continuous, trails_ct_test)
        # lls_ct_svd = ct.soft_clustering(mixture_ct_svd, trails_ct)
        lls_ct_gt_test = ct.soft_clustering(mixture_ct_gt, trails_ct_test)

        lls_ct_em_train = ct.soft_clustering(mixture_ct_em, trails_ct_train)
        lls_ct_kausik_train = ct.soft_clustering(mixture_ct_kausik, trails_ct_train)
        lls_ct_em_continuous_train = ct.soft_clustering(mixture_ct_em_continuous, trails_ct_train)

        em_clustering_error_test = clustering_error(groundtruth_ct_test, lls_ct_em_test, use_max=True)
        kausik_clustering_error_test = clustering_error(groundtruth_ct_test, lls_ct_kausik_test, use_max=True)
        em_continuous_clustering_error_test = clustering_error(groundtruth_ct_test, lls_ct_em_continuous_test, use_max=True)

        em_clustering_error_train = clustering_error(groundtruth_ct_train, lls_ct_em_train, use_max=True)
        kausik_clustering_error_train = clustering_error(groundtruth_ct_train, lls_ct_kausik_train, use_max=True)
        em_continuous_clustering_error_train = clustering_error(groundtruth_ct_train, lls_ct_em_continuous_train, use_max=True)

        em_med_entropy = np.percentile(entropy(lls_ct_em_test, axis=0), 0.5)
        kausik_med_entropy = np.percentile(entropy(lls_ct_kausik_test, axis=0), 0.5)
        gt_med_entropy = np.percentile(entropy(lls_ct_gt_test, axis=0), 0.5)

        # svd_clustering_error = clustering_error(groundtruth, lls_ct_svd)
        print(">>> (TEST) em_clustering_error:", em_clustering_error_test)
        print(">>> (TEST) em_continuous_clustering_error:", em_continuous_clustering_error_test)
        print(f">>> (TEST) entropy: em={em_med_entropy}, gt={gt_med_entropy}")

        import pdb; pdb.set_trace()

        return {
            "em_clustering_error_test": em_clustering_error_test,
            "kausik_clustering_error_test": kausik_clustering_error_test,
            "em_continuous_clustering_error_test": em_continuous_clustering_error_test,
            "em_clustering_error_train": em_clustering_error_train,
            "kausik_clustering_error_train": kausik_clustering_error_train,
            "em_continuous_clustering_error_train": em_continuous_clustering_error_train,
            "em_med_entropy": em_med_entropy,
            "kausik_med_entropy": kausik_med_entropy,
            "gt_med_entropy": gt_med_entropy,
            # "svd_clustering_error": svd_clustering_error,
        }
    
    else:
        assert(False)


@genpath
def plot_lastfm(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_lastfm(row.n, row.L, row.tau, row.t_len, row.L, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("L")

    x = grp["em_clustering_error_test"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    line_solid, = plt.plot(mean.index, mean, color="black", label="test")
    line1, = plt.plot(mean.index, mean, label=label_em, **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    line_dashed, = plt.plot(mean.index, grp["em_clustering_error_train"].mean(), color="black", linestyle="dashed", label="train")
    plt.plot(mean.index, grp["em_clustering_error_train"].mean(), **config, linestyle="dashed")

    x = grp["kausik_clustering_error_test"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    line2, = plt.plot(mean.index, mean, label=label_kausik, **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    plt.plot(mean.index, grp["kausik_clustering_error_train"].mean(), **config, linestyle="dashed")

    next_config()

    x = grp["em_continuous_clustering_error_test"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    line3, = plt.plot(mean.index, mean, label=label_continuous_em, **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    plt.plot(mean.index, grp["em_continuous_clustering_error_train"].mean(), **config, linestyle="dashed")


    legend1 = plt.legend(handles=[line1, line2, line3], loc="lower right")
    plt.gca().add_artist(legend1)
    plt.legend(handles=[line_solid, line_dashed], loc="upper left")
    plt.title(gen_title(setup)) # , {"r": n_observations}))
    plt.xlabel("$L$ (number of users)")
    plt.ylabel("classification error")
    plt.xticks(mean.index)
    savefig()


@genpath
def plot_lastfm_entropy(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_lastfm(row.n, row.L, row.tau, row.t_len, row.L, seed=10+row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("L")

    x = grp["em_med_entropy"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    plt.plot(mean.index, mean, label=label_em, **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["kausik_med_entropy"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    plt.plot(mean.index, mean, label=label_kausik, **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["gt_med_entropy"]
    mean = x.mean()
    std = x.std()
    for _ in range(2): next_config()
    config = next_config()
    plt.plot(mean.index, mean, label="Groundtruth", **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=config["color"])

    x = np.linspace(np.min(mean.index), np.max(mean.index), 100)
    plt.plot(x, np.log(x), label="Max-Entropy", color="black")

    plt.legend(loc="upper left")
    plt.title(gen_title(setup))
    plt.xlabel("$L$ (number of users)")
    plt.ylabel("median entropy")
    plt.xticks(mean.index)
    savefig()


@memoize
def eval_lastfm(n, L, tau, t_len, users, seed=None):
    import LastFM.lastfm as lastfm
    from scipy.stats import entropy

    n_rows = None

    ct_train, ct_test = lastfm.load_ct(
        max_time_delta=lastfm.max_time_delta,
        min_trail_time=lastfm.min_trail_time,
        top_n_songs=n,
        n_rows=n_rows)
    trails_ct, labels_ct = ct_train
    trails_ct_test, labels_ct_test = ct_test

    dt_train, dt_test = lastfm.load_dt(
        max_time_delta=lastfm.max_time_delta,
        min_trail_time=lastfm.min_trail_time,
        top_n_songs=n,
        n_rows=n_rows,
        tau=tau,
        t_len=t_len)
    trails_dt, labels_dt = dt_train

    xs, cs = np.unique(labels_dt, return_counts=True)
    ix = np.argpartition(cs, -users)[-users:]
    max_labels = xs[ix]

    ix_dt = np.isin(labels_dt, max_labels)
    trails_dt = trails_dt[ix_dt]
    labels_dt = labels_dt[ix_dt]

    # unique_labels = np.unique(max_labels)
    # num_unique_labels = len(unique_labels)
    # labels_ix = {label: ix for label, ix in zip(unique_labels, range(num_unique_labels))}

    # trails_ct = [trail for trail, label in zip(trails_ct, labels_ct) if label in max_labels]
    # labels_ct = np.array([labels_ix[label] for label in labels_ct if label in max_labels])

    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    mixture_ct_em = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=100)

    # filter_ct = [len(trail) > 1 for trail in trails_ct]
    # trails_ct = [trail for trail, keep in zip(trails_ct, filter_ct) if keep]

    lls_ct_em = ct.soft_clustering(mixture_ct_em, trails_ct)
    mean_alloc = np.mean(lls_ct_em, axis=1)

    p50, p90, p99 = np.percentile(entropy(lls_ct_em, axis=0), [0.5, 0.9, 0.99])
    amin, amax = np.min(mean_alloc), np.max(mean_alloc)

    return {
        "em_p50": p50,
        "em_p90": p90,
        "em_p99": p99,
        "em_amin": amin,
        "em_amax": amax,
    }



@genpath
def plot_eval_lastfm(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: eval_lastfm(row.n, row.L, row.tau, row.t_len, row.users, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("L")

    # for var in ["em_p50", "em_p90", "em_p99", "em_amin", "em_amax"]:
    x = grp["em_p50"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    plt.plot(mean.index, mean, label="median entropy", **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["em_amin"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    plt.plot(mean.index, mean, label="minimum allocation", **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["em_amax"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    plt.plot(mean.index, mean, label="maximum allocation", **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.title(gen_title(setup))
    plt.xlabel(label_L)
    plt.ylabel("%")
    savefig()


