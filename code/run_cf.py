"""Protocol-sensitivity experiments on full LSNM2024 (packet level, 8 header fields).
Usage: run_exp.py <full.pkl> <fp.pkl> <outdir> <protocol: random|temporal|unseen> <seeds csv> [--quick]"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy.stats import binomtest

FULL, FPP, OUT, PROTO, SEEDS = sys.argv[1:6]
QUICK = "--quick" in sys.argv
os.makedirs(OUT, exist_ok=True)
F = ["Length","IP Length","IP TTL","TCP Length","TCP Window Size","UDP Length","ICMP Type","TCP Destination Port"]
N_TR, N_TE = 10500, 4500
df = pd.read_pickle(FULL)[F+["cls","file_id","Frame Time (Epoch)","row_in_file"]]
df["fp"] = pd.read_pickle(FPP)["fp"].values
le = LabelEncoder().fit(df["cls"].astype(str)); df["y"] = le.transform(df["cls"].astype(str))
K = len(le.classes_)

def make_pools(seed):
    rng = np.random.RandomState(seed)
    if PROTO == "random":
        u = rng.rand(len(df)); return u < 0.7
    if PROTO == "temporal":
        is_tr = np.zeros(len(df), bool)
        for fid, g in df.groupby("file_id"):
            t = g["Frame Time (Epoch)"].fillna(-1).values
            order = np.lexsort((g["row_in_file"].values, t))
            cut = int(0.7 * len(g)); idx = g.index.values[order]
            is_tr[idx[:cut]] = True
        return is_tr
    if PROTO == "unseen":
        ufp = df["fp"].unique()
        tr_fp = set(ufp[rng.rand(len(ufp)) < 0.7])
        return df["fp"].isin(tr_fp).values
    raise ValueError(PROTO)

def sample(mask, n, seed):
    out = []
    for c in range(K):
        idx = np.where(mask & (df["y"].values == c))[0]
        if len(idx) == 0: continue
        rs = np.random.RandomState(seed * 100 + c)
        out.append(rs.choice(idx, size=min(n, len(idx)), replace=False))
    return np.concatenate(out)

def prep(tr, te, cols):
    Xtr = df.loc[tr, cols].copy(); Xte = df.loc[te, cols].copy()
    groups = {"has_ip": "IP Length", "has_tcp": "TCP Length", "has_udp": "UDP Length", "has_icmp": "ICMP Type"}
    for k, c in groups.items():
        Xtr[k] = df.loc[tr, c].notna().astype(float).values; Xte[k] = df.loc[te, c].notna().astype(float).values
    med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), list(Xtr.columns)

def metrics(y, p):
    labs = np.unique(y)
    return {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "f1_macro": f1_score(y, p, labels=labs, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, p, labels=labs, average="weighted", zero_division=0), "n_test_classes": int(len(labs))}

def mk_xgb(seed, n=300):
    return xgb.XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=2,
                             random_state=seed, eval_metric="mlogloss", num_class=K, objective="multi:softprob")

def fit_xgb(model, X, y):
    # remap to contiguous labels in case a class is absent from training (unseen protocol)
    present = np.unique(y); m = {c: i for i, c in enumerate(present)}
    model.set_params(num_class=len(present)); model.fit(X, np.vectorize(m.get)(y))
    return lambda Z: present[model.predict(Z).astype(int)], model, present

def mcnemar(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok))
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only": b, "b_only": c, "p": p}


import shap
from sklearn.model_selection import StratifiedKFold
def encoder(X, y, seed):
    enc = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed); enc.fit(X, y)
    def lat(Z):
        A = Z
        for i in range(len(enc.coefs_) - 1): A = np.maximum(A @ enc.coefs_[i] + enc.intercepts_[i], 0)
        return A
    return lat
for seed in [int(s) for s in SEEDS.split(",")]:
    t0 = time.time(); R = {"protocol": PROTO, "seed": seed}
    pool = make_pools(seed)
    tr = sample(pool, N_TR, seed); te = sample(~pool, N_TE, seed + 7)
    ytr = df["y"].values[tr]; yte = df["y"].values[te]
    trfp = set(df["fp"].values[tr]); seen = np.array([f in trfp for f in df["fp"].values[te]])
    Xtr, Xte, cols = prep(tr, te, F)
    t = time.time(); f, xm, present = fit_xgb(mk_xgb(seed), Xtr, ytr); p_x = f(Xte); tx = time.time() - t
    # split-sample cross-fitted hybrid: encoder on one half, XGBoost on encoded other half, swap, average probabilities
    t = time.time(); skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    probs = np.zeros((len(Xte), K))
    for a, b in skf.split(Xtr, ytr):
        lat = encoder(Xtr[a], ytr[a], seed)
        pres = np.unique(ytr[b]); m = {c: i for i, c in enumerate(pres)}
        xb = mk_xgb(seed); xb.set_params(num_class=len(pres)); xb.fit(lat(Xtr[b]), np.vectorize(m.get)(ytr[b]))
        probs[:, pres] += xb.predict_proba(lat(Xte))
    p_h = probs.argmax(1); th = time.time() - t
    R["models"] = {"xgb": {**metrics(yte, p_x), "train_time_s": round(tx, 2)}, "hybrid_cf": {**metrics(yte, p_h), "train_time_s": round(th, 2)}}
    R["seen_unseen"] = {k: {"acc_seen": float((p[seen] == yte[seen]).mean()) if seen.any() else None,
                            "acc_unseen": float((p[~seen] == yte[~seen]).mean()) if (~seen).any() else None} for k, p in [("xgb", p_x), ("hybrid_cf", p_h)]}
    R["mcnemar_hybridcf_vs_xgb"] = mcnemar(p_h == yte, p_x == yte)
    rs = np.random.RandomState(0); sidx = rs.choice(len(Xte), size=min(2000, len(Xte)), replace=False)
    sv = np.abs(np.array(shap.TreeExplainer(xm).shap_values(Xte[sidx])))
    imp = sv.mean(axis=(0, 2)) if (sv.ndim == 3 and sv.shape[1] == len(cols)) else sv.mean(axis=(0, 1))
    R["shap_mean_abs"] = dict(zip(cols, map(float, imp)))
    R["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(OUT, f"cf_{PROTO}_s{seed}.json"), "w"), indent=1, default=float)
    print(PROTO, seed, "xgb", round(R["models"]["xgb"]["f1_macro"], 4), "hcf", round(R["models"]["hybrid_cf"]["f1_macro"], 4), R["mcnemar_hybridcf_vs_xgb"], "t", R["elapsed_s"], flush=True)
