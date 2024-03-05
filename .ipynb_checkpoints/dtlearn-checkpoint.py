import itertools
import warnings
from dtmixtures import *
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.vq import kmeans2, ClusterError
import time
import cvxpy as cp
from sklearn.cluster import AgglomerativeClustering
import fastcount



def svd_learn_new(sample, n, L=None, compress=True, verbose=None, sample_dist=1, sample_num=None,
                  mixture=None, stats={}, distribution=None, sample_all=False, pair_selection=None,
                  em_refine_max_iter=0, returnsval=False):
    """Learns a discrete-time mixture from a set of 3-trails (cf. CT-SVD in https://arxiv.org/abs/2302.04680)

    sample: (n_samples, 3) array of state observations
    n: number of states
    L: number of chains (if None, the method tries to guess the correct L)
    """

    class PartialR:
        def __init__(self, R, states, i, j):
            self.R = R
            self.Rinv = np.linalg.pinv(R)
            self.states = set(states)
            self.i = i
            self.j = j

        def reconstruct(self):
            Ys = self.R @ Ys_
            Ps = Ys @ Ps_
            S = np.real(np.diagonal(self.R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ self.R.T, axis1=1, axis2=2).T)
            Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
            return Mixture(S, Ms)

    def reconstruct_at(i, j):
        E = ProdsInv[i] @ Prods[j]
        eigs, w = np.linalg.eig(E)
        if L is not None:
            mask = np.argpartition(eigs, -L)[-L:]
        else:
            mask = eigs > 1e-5
        R_ = w[:, mask]
        d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        R = np.diag(d) @ R_.T
        return R

    def combine(parts):
        collectedRinv = np.real(np.vstack(list(p.Rinv.T for p in parts)))
        collectedRinvOrigin = [ i for i, p in enumerate(parts) for _ in range(len(p.Rinv.T)) ]

        # dists = pdist(collectedRinv / np.linalg.norm(collectedRinv, axis=0))
        dists = np.zeros(len(collectedRinv) * (len(collectedRinv) - 1) // 2)
        for k, ((p1, rinv1), (p2, rinv2)) in enumerate(itertools.combinations(zip(collectedRinvOrigin, collectedRinv), 2)):
            dists[k] = np.linalg.norm(rinv1 - rinv2)**2 / (np.linalg.norm(rinv1) * np.linalg.norm(rinv2))
            if p1 == p2:
                dists[k] = 10
            elif parts[p1].i == parts[p2].j or parts[p1].j == parts[p2].i:
                dists[k] /= 10

        # fcluster seems buggy, so here's a quick fix
        dist_mtrx = squareform(dists + 1e-10 * np.random.rand(*dists.shape))
        double_dists = [0 if i//2 == j//2 else dist_mtrx[i//2, j//2]
                        for i, j in itertools.combinations(range(2 * len(dist_mtrx)), 2)]
        lnk = linkage(double_dists, method="complete")
        double_groups = fcluster(lnk, r, criterion="maxclust") - 1
        groups = np.array([g for i, g in enumerate(double_groups) if i%2])
        assert(max(groups)+1 == r)

        combinedRinv = np.zeros((r, r))
        for l in range(r):
            cluster = collectedRinv[groups==l]
            intra_dists = np.sum(squareform(dists)[groups==l][:,groups==l], axis=0)
            center = cluster[np.argmin(intra_dists)]
            combinedRinv[l] = center
            if verbose:
                with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                    avg_dist = np.average(
                        list(np.linalg.norm(x - y, ord=1) for x, y in itertools.combinations(cluster, 2))) if len(
                        cluster) > 1 else 0
                    cen_dist = np.linalg.norm(cluster - center, ord=1, axis=1)
                    print("-" * 10,
                          f"label={l} (size={len(cluster)}, d={avg_dist:.5f}) dist from center: avg={np.average(cen_dist):.5f} max={np.max(cen_dist):.5f}",
                          "-" * 10)
                    print("\n".join([
                        f"{'>' if np.allclose(collectedRinv[i], center) else ' '} {parts[i//L].i:2d} {parts[i//L].j:2d} ({i % L}) {x}"
                        for i, x in
                        zip(np.where(groups==l)[0], str(cluster).split("\n"))]))  # where(labels==l)

        assert(len(combinedRinv) == r)
        R = np.linalg.pinv(combinedRinv.T)

        def asgn_mtrx(mass):
            A = cp.Variable((L, r), boolean=True)
            objective = cp.Minimize(cp.sum(cp.max(A @ mass, axis=0)))
            constraint = cp.sum(A, axis=0) == 1
            prob = cp.Problem(objective, [constraint])
            try:
                # print("<", end="", flush=True)
                prob.solve(verbose=False, solver="CBC", maximumSeconds=5)
                # print(">", end="", flush=True)
                assert(A.value.shape == (L, r))
                return A.value
                # if problem.status == 'optimal': .... else: ....
            except Exception as e:
                print("solver exception:", e)
                return np.tile(np.eye(L), r // L + 1)[:,:r]

        Ys = R @ Ys_
        comp = asgn_mtrx(np.linalg.norm(Ys, axis=2, ord=1).T) if compress else np.eye(r)
        if verbose: print(comp)
        compressedR = comp @ R
        if verbose:
            print(f"compressedR.shape = {compressedR.shape} (ideal is {L},{r})")

        Ys = compressedR @ Ys_
        Zs = compressedR @ Zs_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(Zs @ np.transpose(Ys, axes=(0,2,1)), axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    def em_refine(m, states=range(n), em_refine_max_iter=2):
        states = list(states)
        d = sample #.restrict_to(states)
        # m = mixture.restrict_to(states)
        return em_3learn(d, len(states), L, max_iter=em_refine_max_iter, init_mixture=m)
        """
        all_trail_probs = sample.all_trail_probs()
        states_trail_probs = all_trail_probs[states][:,states][:,:,states]
        states_trail_probs /= np.sum(states_trail_probs)
        d = Distribution.from_all_trail_probs(states_trail_probs)
        return em_3learn(d, len(states), L, max_iter=20, init_mixture=m)
        """

    def find_representative(X):
        X = np.array(X)
        Y = np.empty(X.shape)
        Y[:] = np.nan
        ixs = (X > -0.5) & (X < 1.5)
        Y[ixs] = X[ixs]
        return np.nanmedian(Y, axis=0)

    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
    us, ss, vhs = np.linalg.svd(Os)
    if L is None:
        ss_norm = np.linalg.norm(ss, axis=0)
        for i, s_norm in enumerate(ss_norm):
            stats[f"sval-{i}"] = s_norm
        ratios = ss_norm[:-1] / ss_norm[1:]
        for i, ratio in enumerate(ratios):
            stats[f"sval-ratio-{i}"] = ratio
        L = 1 + np.argmax(ratios * (ss_norm[:-1] > 1e-6))
        stats["guessedL"] = L
        # import pdb; pdb.set_trace()
        if mixture is not None:
            sigma_min = min(np.min(np.linalg.svd(X, compute_uv=False)) for i in range(n) for X in [mixture.Ms[:,i,:], mixture.Ms[:,:,i]])
            stats["sigma_min"] = sigma_min
    Ps_ = np.moveaxis(us, 1, 2)[:,:L]
    Qs_ = (ss[:,:L].T * vhs[:,:L].T).T

    A = np.zeros((2 * n * L, n ** 2))
    for j in range(n):
        A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
        A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    s_inc = s[::-1]
    s_inc_c = s_inc[:n-1]
    ratios = s_inc_c[1:] / s_inc_c[:-1]
    r = max(L, 1 + np.argmax(ratios * (s_inc_c[1:] < 0.1)))
    # import pdb; pdb.set_trace()
    if returnsval: return s_inc
    if verbose:
        print(s_inc[:4])
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print("singular values of A:", " ".join( f"{x:.5f}" for x in s_inc))
            print("ratios:                      ", " ".join( f"{x:.5f}" for x in ratios))
            print(f"suggest r={r} (vs L={L})")
    B = vh[-r:] # np.random.rand(r,r) @
    Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    Prods = Zs_ @ np.transpose(Ys_, axes=(0,2,1))
    ProdsInv = np.linalg.pinv(Prods)

    dists = []
    dists2 = []
    for i, j in itertools.combinations(range(n), 2):
        E = ProdsInv[i] @ Prods[j]
        Einv = ProdsInv[j] @ Prods[i]
        # randomized pseudoinverse test
        x = E @ np.random.rand(r, 1000)
        y = Einv @ E @ x
        dists.append(np.linalg.norm(x - y)**2 / (np.linalg.norm(x) * np.linalg.norm(y)))
        dists2.append(np.linalg.norm(x - y))

    lnk = linkage(np.array(dists), method="complete")
    groups = fcluster(lnk, sample_dist, criterion="distance") if sample_num is None else \
        fcluster(lnk, sample_num, criterion="maxclust")

    if verbose:
        for t in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
            groups_ = fcluster(lnk, t, criterion="distance")
            if verbose: print(groups_, t)

    dist_mtrx = squareform(dists)
    np.fill_diagonal(dist_mtrx, np.inf)
    parts = []

    if True:
        indMs = np.zeros((L, n, n))
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            group_dists = dist_mtrx[states][:,states]
            for i in states:
                j = np.argmin(dist_mtrx[i])
                R = reconstruct_at(i, j)
                p = PartialR(R, states, i, j)
                m = p.reconstruct()
                indMs[:,i] = m.Ms[:,i]
        if verbose and mixture is not None:
            print(states, "sample_ind:", Mixture.perm_dist(indMs[:, states], mixture.Ms[:, states]) / n)

    """
    if True: # if sample_all:
        Ms = np.zeros((L, n, n))
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            colMs, colS = [], []
            for i, j in itertools.combinations(states, 2):
                R = reconstruct_at(i, j)
                p = PartialR(R, states, i, j)
                m = p.reconstruct()
                colMs.append(m.Ms[:,states])
                colS.append(m.S[:,states])
            comMs = np.median(colMs, axis=0)
            comMs2 = find_representative(colMs)
            comS = np.median(colMs, axis=0)
            if verbose:
                print(states, "sample_all (median):   ", Mixture.perm_dist(comMs[:, states], mixture.Ms[:, states]) / n)
                print(states, "sample_all (find_repr):", Mixture.perm_dist(comMs2[:, states], mixture.Ms[:, states]) / n)
    """

    if True: # else:
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            group_dists = dist_mtrx[states][:,states]
            if len(states) > 1:
                a, b = np.unravel_index(np.argmin(group_dists), group_dists.shape)
                i, j = states[a], states[b]
                if pair_selection == "best":
                    best_pair = None
                    best_dist = np.inf
                    partial_sample = sample.restrict_to(states)
                    for i, j in itertools.combinations(states, 2):
                        R = reconstruct_at(i, j)
                        p = PartialR(R, states, i, j)
                        m = p.reconstruct().restrict_to(states)
                        m = Mixture(np.abs(m.S), np.abs(m.Ms))
                        m.normalize()
                        partial_distribution = Distribution.from_mixture(m, 3)
                        dist = partial_sample.dist(partial_distribution)
                        # print(i, j, dist)
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (i, j)
                    i, j = best_pair
                elif pair_selection == "rnd":
                    i, j = np.random.choice(states, 2, replace=False)
                elif pair_selection == "worst":
                    a, b = np.unravel_index(np.argmax(group_dists), group_dists.shape)
                    i, j = states[a], states[b]
            else:
                [i] = states
                j = np.argmin(dist_mtrx[i])
            if verbose: print(f"=== group {g}: {states} i={i} j={j} {'='*50}")
            R = reconstruct_at(i, j)
            parts.append(PartialR(R, states, i, j))

        if len(parts) > 1:
            S, Ms = combine(parts)
        else:
            m = parts[0].reconstruct()
            S, Ms = m.S, m.Ms

    if mixture is not None and verbose:
        for p in parts:
            p_mixture = p.reconstruct()
            print(p.states, "combined Ms:", Mixture.perm_dist(Ms[:, list(p.states)], mixture.Ms[:, list(p.states)]) / n,
                  "[vs] part Ms:", Mixture.perm_dist(p_mixture.Ms[:, list(p.states)], mixture.Ms[:, list(p.states)]) / n)
            # print(p.states, f"combined Ms:", np.linalg.norm((Ms - mixture.Ms)[:, list(p.states)].flatten(), ord=1) / f,
            #       "[vs] part Ms:", np.linalg.norm((p_mixture.Ms - mixture.Ms)[:, list(p.states)].flatten(), ord=1) / f)
        """
        print("normalization:")
        m1 = Mixture(S.copy(), Ms.copy())
        m1_ = Mixture(S.copy(), Ms.copy())
        m1_.normalize()
        m2 = Mixture(np.abs(S), np.abs(Ms))
        m2_ = Mixture(np.abs(S), np.abs(Ms))
        m2_.normalize()
        m3 = Mixture(np.abs(S), np.abs(Ms))
        m3_ = Mixture(np.clip(S, 0, 1), np.clip(Ms, 0, 1))
        m3_.normalize()
        m4 = Mixture(np.abs(S), np.abs(Ms))
        m4.normalize()
        m4_ = em_refine(m4, p.states)
        print(np.round(np.sum(m1.Ms, axis=2), 4))
        print(np.round(np.sum(m2.Ms, axis=2), 4))
        print(np.round(np.sum(m3.Ms, axis=2), 4))
        print("      () recov-err:", Mixture.recovery_error(m1, mixture))
        print("     (n) recov-err:", Mixture.recovery_error(m1_, mixture))
        print("   (abs) reocv-err:", Mixture.recovery_error(m2, mixture))
        print(" (abs,n) reocv-err:", Mixture.recovery_error(m2_, mixture))
        print("(abs,em) recov-err:", Mixture.recovery_error(m4_, mixture))
        # print("     (0) reocv-err:", Mixture.perm_dist(m3.Ms, mixture.Ms) / n)
        # print("   (0,n) reocv-err:", Mixture.perm_dist(m3_.Ms, mixture.Ms) / n)
        print("   (abs) tv_dist:  ", distribution.dist(Distribution.from_mixture(m2, 3)))
        print(" (abs,n) tv_dist:  ", distribution.dist(Distribution.from_mixture(m2_, 3)))
        # print("     (0) tv_dist:  ", distribution.dist(Distribution.from_mixture(m3, 3)))
        # print("   (0,n) tv_dist:  ", distribution.dist(Distribution.from_mixture(m3_, 3)))
        print("(abs,em) tv_dist:  ", distribution.dist(Distribution.from_mixture(m4_, 3)))
        """

    S, Ms = np.abs(S), np.abs(Ms)
    learned_mixture = Mixture(S, Ms)
    if em_refine_max_iter > 0:
        learned_mixture = em_refine(learned_mixture, em_refine_max_iter=em_refine_max_iter)
    learned_mixture.normalize()

    return learned_mixture


def svd_learn(sample, n, L=None, sv_threshold=0.05, verbose=None, stats={}, mixture=None, Os=None):
    """Learns a discrete-time mixture from a set of 3-trails (cf. https://theory.stanford.edu/~sergei/papers/nips16-mcc.pdf)

    sample: (n_samples, 3) array of state observations
    n: number of states
    L: number of chains
    """
    # assert(len(sample.trails) == n**sample.t_len)
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0) if Os is None else Os

    svds = [ np.linalg.svd(Os[j], full_matrices=True) for j in range(n) ]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    if L is None:
        above_thresh = [ np.sum(s / s[0] > sv_threshold) for _, s, _ in svds ]
        L = int(np.median(above_thresh))
        if verbose is not None:
            for (_, s, _), t in zip(svds, above_thresh):
                print(s / s[0], t)
            print("Guessed L={}".format(L))

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    small = list(s < 1e-5)
    if True in small:
        fst = small.index(True)
        if verbose: print(2*L*n - fst, L, s[[fst-1, fst]])
    B = vh[-L:]
    Bre = np.moveaxis(B.reshape((L, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    if verbose:
        print([ np.linalg.matrix_rank(Zs_[j] @ Ys_[j].T) for j in range(n) ])
        print([ np.linalg.svd(Zs_[j] @ Ys_[j].T)[1] for j in range(n) ])
    # Xs = [ np.linalg.inv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    Xs = [ np.linalg.pinv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    X = np.sum(Xs, axis=0)
    # X = np.linalg.inv(Zs_[0] @ Ys_[0].T) @ (Zs_[1] @ Ys_[1].T)
    _, R_ = np.linalg.eig(X)
    d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[0] @ Ps_[0]).T, Os[0] @ np.ones(n), rcond=None)
    # maybe average over d, too?

    R = np.diag(d) @ R_.T
    Ys = R @ Ys_

    Ps = np.array([ Y @ P_ for Y, P_ in zip(Ys, Ps_) ])
    Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])
    # print(Ss)

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))
    for l in range(L):
        for i in range(n):
            S_[l,i] = Ss[i,l,l]
            for j in range(n):
                # only good if warning "Casting complex values to real discards the imaginary part" occurs here:
                Ms_[l,i,j] = Ps[j,l,i] / S_[l,i]

    # S_ = np.clip(np.real(S_), 0, 1)
    # Ms_ = np.clip(np.real(Ms_), 0, 1)
    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    if np.isnan(S_).any() or np.isnan(Ms_).any():
        print("SVD failed! Returning random mixture.")
        return Mixture.random(n, L)
    learned_mixture = Mixture(S_, Ms_)
    if verbose: print(learned_mixture)
    learned_mixture.normalize()
    # print(">>>", learned_mixture)
    return learned_mixture


def em_3learn(sample, n, L, max_iter=1000, ll_stop=1e-4, verbose=None, init_mixture=None, stats={}, mixture=None,
             write_stats=False):
    """Learns a discrete-time mixture from a set of 3-trails using expectation maximization (EM).

    sample: (n_samples, 3) array of state observations
    n: number of states
    L: number of chains
    """
    flat_mixture = (init_mixture or Mixture.random(n, L)).flat()
    flat_trails, trail_probs = sample.flat_trails()

    prev_lls = 0
    for n_iter in range(max_iter):
        lls = flat_trails @ np.log(flat_mixture + 1e-20).transpose()
        if ll_stop is not None and np.max(np.abs(prev_lls - lls)) < ll_stop: break
        prev_lls = lls

        raw_probs = np.exp(lls)
        cond_probs = raw_probs / np.sum(raw_probs, axis=1)[:, np.newaxis]
        cond_probs[np.any(np.isnan(cond_probs), axis=1)] = 1 / L

        flat_mixture = cond_probs.transpose() @ (flat_trails * trail_probs[:, np.newaxis])
        # normalize:
        flat_mixture[:, :n] /= np.sum(flat_mixture[:, :n])
        for i in range(n):
            rows = flat_mixture[:, (i+1)*n:(i+2)*n]
            rows[:] = rows / rows.sum(axis=1)[:, np.newaxis]
            rows[np.any(np.isnan(rows), axis=1)] = 1 / n

        if verbose is not None:
            learned_mixture = Mixture.from_flat(flat_mixture, n)
            learned_distribution = Distribution.from_mixture(learned_mixture, sample.t_len)
            print("Iteration {}: recovery_error={} tv_dist={}".format(
                n_iter + 1, verbose.recovery_error(learned_mixture) if isinstance(verbose, Mixture) else np.inf,
                learned_distribution.dist(sample)))

    if write_stats: stats["n_iter"] = n_iter
    return Mixture.from_flat(flat_mixture, n)


def kausik_cluster(n, L, trails):
    n_samples, t_len = trails.shape
    segment_len = t_len // 4

    # subspace estimation
    M = np.zeros((n, n, n))
    next_state = np.zeros((n, n_samples, 2, n))
    for i in range(n):
        for t, trail in enumerate(trails):
            for segno, trail_segment in enumerate([trail[segment_len:2*segment_len], trail[3*segment_len:4*segment_len]]):
                states, counts = np.unique(trail_segment[1:][trail_segment[:-1] == i], return_counts=True)
                next_state[i, t, segno,states] = counts / np.sum(counts)
            M[i] += np.outer(next_state[i, t, 0], next_state[i, t, 1])
    M /= n_samples
    X = M + np.moveaxis(M, 1, 2)
    u, s, vh = np.linalg.svd(X)
    V = np.zeros((n, n, n))
    for i in range(n):
        V[i] = u[i,:,:L] @ np.diag(s[i,:L]) @ vh[i,:L,:]

    # clustering
    dist = np.zeros((n_samples, n_samples))
    for t1, t2 in itertools.product(range(n_samples), repeat=2):
        D = np.zeros((2, n, n))
        for i in range(n):
            for segno in range(2):
                D[segno, i] = V[i] @ (next_state[i, t1, segno] - next_state[i, t2, segno])
        dist[t1, t2] = np.max(np.inner(D[0], D[1]))
    clustering = AgglomerativeClustering(metric="precomputed", linkage="complete", n_clusters=L).fit(dist)
    labels = clustering.labels_
    # print(labels)

    return labels


def kausik_learn(n, L, trails, learn=True):
    """Learns a discrete-time mixture from a set of (long) arbitrary-length trails (cf. https://arxiv.org/abs/2211.09403)
    
    n: number of states
    L: number of clusters
    trails: set of trails of arbitrary length
    learn: if True, learns the a mixture, otherwise returns the clustered trails
    """

    labels = kausik_cluster(n, L, trails)

    if learn:
        # learning
        S = np.zeros((L, n))
        Ms = np.zeros((L, n, n))
        for l in range(L):
            chain_trails = trails[labels == l]
            states, counts = np.unique(chain_trails[:, 0], return_counts=True)
            S[l, states] = counts / np.sum(counts)
            for i in range(n):
                states, counts = np.unique(chain_trails[:,1:][chain_trails[:,:-1] == i], return_counts=True)
                Ms[l,i,states] = counts / np.sum(counts)

        return Mixture(S, Ms)

    else:
        # return clustered trails
        return list([trails[labels == l] for l in range(L)])



def likelihood(mixture, trails, counts=None, log=False):
    if counts is None: counts = fastcount.transitions(mixture.n, trails)
    logS = np.log(mixture.S + 1e-10)
    logTs = np.log(mixture.Ms + 1e-10)

    logl = logS[:, trails[:,0]]
    logl += np.sum(logTs[:, :, :, None] * np.moveaxis(counts, 0, 2)[None, :, :, :], axis=(1,2))
    if log: return logl
    probs = np.exp(logl - np.max(logl, axis=0))
    probs /= np.sum(probs, axis=0)[None, :]
    return probs


def mle(n, trails, ll, counts=None, starts=None):
    if counts is None: counts = fastcount.transitions(n, trails)
    if starts is None: starts = trails[:,0][:,None] == np.arange(n)[None,:]
    L = len(ll)
    S = np.zeros((L, n))
    Ms = np.zeros((L, n, n))
    for l in range(L):
        S[l] = np.sum(ll[l][:, None] * starts, axis=0)
        Ms[l] = np.sum(ll[l][:, None, None] * counts, axis=0)
    mixture = Mixture(S + 1e-10, Ms + 1e-10)
    mixture.normalize()
    return mixture


def em_learn(n, L, trails, max_iter=100, init=None, conv_thresh=1e-5, verbose=False):
    counts = fastcount.transitions(n, trails)
    starts = trails[:,0][:,None] == np.arange(n)[None,:]
    mixture = Mixture.random(n, L) if init is None else init
    prev_ll = None

    for iter in range(max_iter):
        ll = likelihood(mixture, trails, counts)
        mixture = mle(n, trails, ll, counts, starts)

        linf = 1 if prev_ll is None else np.max(np.abs(ll - prev_ll))
        if verbose: print(f"Iteration {iter+1}/{max_iter}: {linf}")
        if linf < conv_thresh: break
        prev_ll = ll

    return mixture







#   def em_long_trails(n, L, trails, n_iter=100):
#       """Learns a discrete-time mixture from a set of trails of arbitrary length"""
#       n_samples, t_len = trails.shape
#       mixture = Mixture.random(n, L)
#       eps = 1e-10
#
#       for _ in range(n_iter):
#           logS = np.log(mixture.S + eps)
#           logMs = np.log(mixture.Ms)
#
#           logl = logS[:, trails[:,0]] # L x n_samples
#           for i in range(1, t_len):
#               logl += logMs[:, trails[:,i-1], trails[:,i]]
#           probs = np.exp(logl - np.max(logl, axis=0))
#           probs /= np.sum(probs, axis=0)[None, :]
#
#           for l in range(L):
#               mixture.S[l] = probs[l] @ (trails[:,0][:,None] == np.arange(n)[None,:])
#               next_state = probs[l][:,None,None] * (trails[:,1:][:,:,None] == np.arange(n)[None,None,:])
#               for i in range(n):
#                   X = (trails[:,:-1] == i)[:,:,None] * next_state
#                   mixture.Ms[l,i,:] = np.sum(X, axis=(0,1))
#           mixture.S += eps
#           mixture.Ms += eps
#           mixture.normalize()
#
#       return mixture


