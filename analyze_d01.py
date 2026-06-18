#!/usr/bin/env python
"""
D0.1 taxonomy-signal analysis at high N.

Inputs (--dir): evo2_block26_mean.npy + evo2_meta.csv (accession order),
optionally esm2_3b_mean.npy + esm_meta.csv, and manifest.csv (labels).

Reports, for EVO2 / ESM / concat:
  * global Mantel r (embedding cos-dist ~ taxonomy rank-dist), + partial | log10(len)
  * per-family Mantel (families with n>=MINF)
  * per-novelty-tier Mantel (in_distribution vs derived_eukaryotic)
  * family classification: 5-fold stratified CV accuracy vs chance, kNN purity
Mantel uses a subsample (default 1500 taxa) for the permutation test to stay fast.
"""
import os, sys, json, argparse, numpy as np, csv
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--sub", type=int, default=1500)   # taxa for Mantel permutation
ap.add_argument("--perms", type=int, default=299)
ap.add_argument("--minf", type=int, default=20)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# ---- labels ----------------------------------------------------------------
man = {}
with open(args.manifest) as f:
    for r in csv.DictReader(f):
        man[r["accession"]] = r
RANKS = ["genus", "family", "order", "class", "phylum", "kingdom"]

def load(name_npy, name_meta):
    p = os.path.join(args.dir, name_npy)
    if not os.path.exists(p):
        return None, None
    X = np.load(p)
    accs = []
    with open(os.path.join(args.dir, name_meta)) as f:
        rd = csv.DictReader(f)
        for row in rd:
            accs.append(row["accession"])
    accs = accs[: len(X)]
    X = X[: len(accs)]
    return X, accs

EV, ev_acc = load("evo2_block26_mean.npy", "evo2_meta.csv")
ES, es_acc = load("esm2_3b_mean.npy", "esm_meta.csv")

def align(X, accs):
    keep = [i for i, a in enumerate(accs) if a in man and man[a].get("family")]
    return X[keep], [accs[i] for i in keep]

def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

mats = {}
if EV is not None:
    mats["EVO2"] = align(EV, ev_acc)
if ES is not None:
    mats["ESM2-3B"] = align(ES, es_acc)
if EV is not None and ES is not None:
    # L2-normalised concat on shared, family-labelled accessions
    s = {a: i for i, a in enumerate(ev_acc)}
    t = {a: i for i, a in enumerate(es_acc)}
    common = [a for a in ev_acc if a in t and a in man and man[a].get("family")]
    C = np.concatenate([_l2(EV[[s[a] for a in common]].astype(np.float64)),
                        _l2(ES[[t[a] for a in common]].astype(np.float64))], axis=1)
    mats["EVO2+ESM2"] = (C, common)

def cos_dist(X):
    Xn = _l2(X.astype(np.float64))
    S = Xn @ Xn.T
    return 1.0 - S

def tax_dist(accs):
    n = len(accs)
    codes = np.zeros((n, len(RANKS)), dtype=object)
    for i, a in enumerate(accs):
        for k, rk in enumerate(RANKS):
            codes[i, k] = man[a].get(rk, "") or "__%d" % i  # empty -> unique
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k, rk in enumerate(RANKS):
            same = codes[:, k] == codes[i, k]
            # distance contribution: ranks at which they still differ
            D[i] += (~same).astype(np.float64)
    np.fill_diagonal(D, 0)
    return D  # 0..len(RANKS); higher = more distant

def utri(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]

def rank_matrix(D):
    """Symmetric matrix whose upper-tri entries are ranked once (Spearman prep).
    Lets permutations reindex precomputed ranks instead of calling rankdata each time."""
    from scipy.stats import rankdata
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    r = rankdata(D[iu])
    R = np.zeros((n, n))
    R[iu] = r
    R[(iu[1], iu[0])] = r
    return R

def mantel(Dx, Dy, perms, rng):
    n = Dx.shape[0]; iu = np.triu_indices(n, 1)
    rx = rank_matrix(Dx)[iu]
    RY = rank_matrix(Dy); ry = RY[iu]
    r = np.corrcoef(rx, ry)[0, 1]
    cnt = 1
    for _ in range(perms):
        p = rng.permutation(n)
        if np.corrcoef(rx, RY[np.ix_(p, p)][iu])[0, 1] >= r:
            cnt += 1
    return r, cnt / (perms + 1)

def partial_mantel(Dx, Dy, Dz, perms, rng):
    n = Dx.shape[0]; iu = np.triu_indices(n, 1)
    RX = rank_matrix(Dx); rx = RX[iu]
    ry = rank_matrix(Dy)[iu]; rz = rank_matrix(Dz)[iu]
    def pcor(a, b, c):
        rab = np.corrcoef(a, b)[0, 1]; rac = np.corrcoef(a, c)[0, 1]; rbc = np.corrcoef(b, c)[0, 1]
        return (rab - rac * rbc) / (np.sqrt(1 - rac**2) * np.sqrt(1 - rbc**2) + 1e-12)
    r = pcor(rx, ry, rz); cnt = 1
    for _ in range(perms):
        p = rng.permutation(n)
        if pcor(RX[np.ix_(p, p)][iu], ry, rz) >= r:
            cnt += 1
    return r, cnt / (perms + 1)

def lendist(accs):
    L = np.array([np.log10(float(man[a].get("length_nt", 1)) + 1) for a in accs])
    return np.abs(L[:, None] - L[None, :])

def subsample(accs, k, rng):
    idx = np.arange(len(accs))
    if len(idx) <= k:
        return idx
    return np.sort(rng.choice(idx, k, replace=False))

def probe(X, y):
    # fast cosine-kNN family classification, 5-fold CV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    c = Counter(y); keep = [i for i, l in enumerate(y) if c[l] >= 5]
    Xk = X[keep].astype(np.float32); yk = np.array(y)[keep]
    Xn = Xk / (np.linalg.norm(Xk, axis=1, keepdims=True) + 1e-9)
    knn = KNeighborsClassifier(n_neighbors=15, metric="cosine")
    sc = cross_val_score(knn, Xn, yk, cv=5)
    p = np.array(list(Counter(yk).values())) / len(yk)
    return float(sc.mean()), float(sc.std()), float((p**2).sum()), len(yk), len(set(yk))

print("=" * 70)
for name, (X, accs) in mats.items():
    print(f"\n### {name}  (N={len(accs)})")
    fams = [man[a]["family"] for a in accs]
    tiers = [man[a].get("novelty_tier", "") for a in accs]

    # ---- global Mantel on a subsample
    sidx = subsample(accs, args.sub, rng)
    sa = [accs[i] for i in sidx]
    Dx = cos_dist(X[sidx]); Dy = tax_dist(sa); Dz = lendist(sa)
    r, p = mantel(Dx, Dy, args.perms, rng)
    pr, pp = partial_mantel(Dx, Dy, Dz, args.perms, rng)
    rl, pl = mantel(Dz, Dy, args.perms, rng)
    print(f"  Mantel  tax~emb         r={r:+.3f} p={p:.4f}   (sub n={len(sa)})")
    print(f"  partial tax~emb|len     r={pr:+.3f} p={pp:.4f}")
    print(f"  Mantel  tax~length      r={rl:+.3f} p={pl:.4f}   (the confound)")

    # ---- per novelty tier
    for tier in ["in_distribution", "lenarvirus_eukaryotic", "derived_eukaryotic"]:
        ti = [i for i in range(len(accs)) if tiers[i] == tier]
        if len(ti) < 30:
            continue
        ti = subsample([accs[i] for i in ti], min(args.sub, len(ti)), rng) if len(ti) > args.sub else np.array(ti)
        ta = [accs[i] for i in ti]
        rr, ppv = mantel(cos_dist(X[ti]), tax_dist(ta), 99, rng)
        print(f"  tier {tier:24s} Mantel r={rr:+.3f} p={ppv:.3f}  n={len(ta)}")

    # ---- per family (n>=minf): does within-family structure exist?
    print(f"  per-family Mantel (n>={args.minf}):")
    fam_counts = Counter(fams)
    for fam, c in sorted(fam_counts.items(), key=lambda x: -x[1]):
        if c < args.minf:
            continue
        fi = [i for i in range(len(accs)) if fams[i] == fam]
        fa = [accs[i] for i in fi]
        # within a family, taxonomy distance uses genus/species ranks
        rr, ppv = mantel(cos_dist(X[fi]), tax_dist(fa), 99, rng)
        print(f"     {fam:22s} n={c:4d}  r={rr:+.3f} p={ppv:.3f}")

    # ---- classification probe
    acc, sd, chance, nk, nc = probe(X, fams)
    print(f"  family probe (5-fold): acc={acc:.3f}±{sd:.3f}  chance={chance:.3f}  ({nk} seqs, {nc} families)")

print("\n" + "=" * 70)
print("done")
