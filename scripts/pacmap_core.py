import numpy as np
import pickle
import time
import os

import pacmap
import pacmap_source as _pm


def load_mammoth(path="data/mammoth_3d.npy"):
    if not os.path.exists(path):
        import urllib.request, json as _json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading Mammoth dataset...")
        url = ("https://raw.githubusercontent.com/PAIR-code/"
               "understanding-umap/master/raw_data/mammoth_3d.json")
        json_path = path.replace(".npy", ".json")
        urllib.request.urlretrieve(url, json_path)
        X = np.array(_json.load(open(json_path)))
        np.save(path, X)
    X = np.load(path)
    return X, X[:, 0]

def default_schedule(w_MN_init, itr, *, num_iters):
    p1, p2, _ = num_iters
    if itr < p1:
        return (1 - itr/p1)*w_MN_init + (itr/p1)*3.0, 2.0, 1.0
    elif itr < p1 + p2:
        return 3.0, 3.0, 1.0
    return 0.0, 1.0, 1.0

def constant_low(w_MN_init, itr, *, num_iters):
    return 3.0, 3.0, 1.0

def constant_zero_mn(w_MN_init, itr, *, num_iters):
    return 0.0, 1.0, 1.0

def no_phase1(w_MN_init, itr, *, num_iters):
    p1, p2, _ = num_iters
    return (3.0, 3.0, 1.0) if itr < p1 + p2 else (0.0, 1.0, 1.0)

def no_phase3(w_MN_init, itr, *, num_iters):
    p1, _, _ = num_iters
    if itr < p1:
        return (1 - itr/p1)*w_MN_init + (itr/p1)*3.0, 2.0, 1.0
    return 3.0, 3.0, 1.0

def reversed_schedule(w_MN_init, itr, *, num_iters):
    total = sum(num_iters)
    t1, t2 = int(total * 0.55), int(total * 0.78)
    if itr < t1:
        return 0.0, 1.0, 1.0
    elif itr < t2:
        return 3.0, 3.0, 1.0
    p = (itr - t2) / (total - t2)
    return 3.0*(1-p) + w_MN_init*p, 2.0, 1.0

SCHEDULES = {
    "default": default_schedule, "constant_low": constant_low,
    "constant_zero_mn": constant_zero_mn, "no_phase1": no_phase1,
    "no_phase3": no_phase3, "reversed": reversed_schedule,
}

def make_schedule(config):
    """Build schedule: base schedule + pair-type weight masking."""
    # Start with base schedule
    base = SCHEDULES[config["schedule"]]
    # Apply pair-type weight masking if specified in config
    a = config.get("active_pairs", {"near": True, "mid_near": True, "further": True})
    # If all pair types are active, return the base schedule directly
    if all(a.get(k, True) for k in ("near", "mid_near", "further")):
        return base
    # Otherwise, return a masked version of the base schedule
    nb, mn, fp = a.get("near", True), a.get("mid_near", True), a.get("further", True)
    def masked(w_MN_init, itr, *, num_iters):
        w_MN, w_NB, w_FP = base(w_MN_init, itr, num_iters=num_iters)
        return w_MN if mn else 0.0, w_NB if nb else 0.0, w_FP if fp else 0.0
    return masked

def triplet_accuracy(X, Y, n=50000, seed=42):
    """Estimate triplet accuracy: P(dh_ij < dh_ik) == P(dl_ij < dl_ik)."""
    rng = np.random.RandomState(seed)
    # Sample random triplets (i,j,k) and check if the relative distances are preserved
    idx = rng.randint(0, len(X), (n, 3))
    i, j, k = idx[:,0], idx[:,1], idx[:,2]
    # Compute distances in original and embedding spaces for the triplets
    dh_ij = np.sum((X[i]-X[j])**2, 1); dh_ik = np.sum((X[i]-X[k])**2, 1)
    dl_ij = np.sum((Y[i]-Y[j])**2, 1); dl_ik = np.sum((Y[i]-Y[k])**2, 1)
    # Only consider valid triplets where the original distances are not equal (to avoid ties)
    valid = dh_ij != dh_ik
    return float(((dh_ij < dh_ik) == (dl_ij < dl_ik))[valid].mean())

def neighbor_preservation(X, Y, k=10):
    """Estimate neighbor preservation: P(nh[i] == nl[i]) for i in range(len(X)))."""
    from sklearn.neighbors import NearestNeighbors
    # Compute k-nearest neighbors in original and embedding spaces, then average the overlap
    nh = NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(X, return_distance=False)[:,1:]
    nl = NearestNeighbors(n_neighbors=k+1).fit(Y).kneighbors(Y, return_distance=False)[:,1:]
    return float(np.mean([len(set(nh[i]) & set(nl[i]))/k for i in range(len(X))]))

def compute_metrics(X, Y):
    return {
        "triplet_acc": triplet_accuracy(X, Y),
        "neighbor_pres": neighbor_preservation(X, Y)
        }

def run_pacmap(X, config):
    """Run PaCMAP with a custom weight schedule and intermediate snapshots."""
    SNAP_ITERS = [0, 10, 50, 100, 200, 450]
    init = config.get("init", "pca")
    
    # Patch the weight schedule into pacmap's internal function
    original = _pm.find_weight
    _pm.find_weight = make_schedule(config)
    t0 = time.time()

    try:
        # Use pip pacmap for pair generation and preprocessing
        model = pacmap.PaCMAP(
            n_components=2, n_neighbors=10, MN_ratio=0.5,
            FP_ratio=2.0, random_state=42, num_iters=450,
        )
        # Preprocess and generate pairs via the standard API
        X_proc = np.copy(X).astype(np.float32)
        n, dim = X_proc.shape
        X_proc, pca_solution, tsvd, model.xmin, model.xmax, model.xmean = \
            _pm.preprocess_X(X_proc, model.distance, model.apply_pca,
                             model.verbose, model.random_state, dim, model.n_components)
        model.tsvd_transformer = tsvd
        model.pca_solution = pca_solution
        model.decide_num_pairs(n)
        model.sample_pairs(X_proc, model.save_tree)

        # Run optimization 
        Y, intermediate_states, _, _, _ = _pm.pacmap(
            X_proc,
            model.n_components,
            model.pair_neighbors,
            model.pair_MN,
            model.pair_FP,
            model.lr,
            model.num_iters,
            init,
            model.verbose,
            True,              # intermediate=True
            SNAP_ITERS,        # our snapshot iterations
            pca_solution,
            tsvd
        )

        # Build snapshots dict from intermediate_states
        snapshots = {}
        for idx, t in enumerate(SNAP_ITERS):
            if idx < intermediate_states.shape[0]:
                snapshots[t] = intermediate_states[idx].copy()

    finally:
        # Restore original function to avoid side effects on other runs
        _pm.find_weight = original

    elapsed = time.time() - t0
    metrics = [{"iter": t, **compute_metrics(X, Y_snap)} for t, Y_snap in snapshots.items()]
    for e in metrics:
        print(f"    iter {e['iter']:>4d} | triplet={e['triplet_acc']:.3f} | neighbor={e['neighbor_pres']:.3f}")
    print(f"  Done in {elapsed:.1f}s")

    return {"name": config["name"], "config": config, "time": elapsed,
            "embedding": snapshots[SNAP_ITERS[-1]], "snapshots": snapshots, "metrics": metrics}

def save_result(r, d="results"):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{r['name']}.pkl"), "wb") as f: pickle.dump(r, f)

def load_result(name, d="results"):
    with open(os.path.join(d, f"{name}.pkl"), "rb") as f: return pickle.load(f)