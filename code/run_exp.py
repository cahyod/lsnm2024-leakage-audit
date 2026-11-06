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

for seed in [int(s) for s in SEEDS.split(",")]:
    t0 = time.time(); R = {"protocol": PROTO, "seed": seed}
    pool = make_pools(seed)
    tr = sample(pool, N_TR, seed); te = sample(~pool, N_TE, seed + 7)
    ytr = df["y"].values[tr]; yte = df["y"].values[te]
    R["n_train"] = int(len(tr)); R["n_test"] = int(len(te))
    R["train_per_class"] = np.bincount(ytr, minlength=K).tolist(); R["test_per_class"] = np.bincount(yte, minlength=K).tolist()
    trfp = set(df["fp"].values[tr]); seen = np.array([f in trfp for f in df["fp"].values[te]])
    R["test_seen_frac"] = float(seen.mean())
    # pool-level overlap (whole partitions, not just samples)
    pool_tr_fp = set(df["fp"].values[pool]); R["pool_test_seen_frac"] = float(df["fp"][~pool].isin(pool_tr_fp).mean())
    Xtr, Xte, cols = prep(tr, te, F)
    preds = {}; times = {}
    # 1 memoriser (lookup table of training fingerprints; unseen -> Benign)
    t = time.time(); tab = pd.DataFrame({"fp": df["fp"].values[tr], "y": ytr}).groupby("fp")["y"].agg(lambda s: s.value_counts().idxmax())
    fb = int(le.transform(["Benign"])[0])
    preds["memoriser"] = pd.Series(df["fp"].values[te]).map(tab).fillna(fb).astype(int).values; times["memoriser"] = time.time() - t
    # 2 logistic regression
    t = time.time(); m = LogisticRegression(max_iter=300 if QUICK else 1000); m.fit(Xtr, ytr); preds["logreg"] = m.predict(Xte); times["logreg"] = time.time() - t
    # 3 random forest
    t = time.time(); m = RandomForestClassifier(n_estimators=100 if QUICK else 200, n_jobs=2, random_state=seed); m.fit(Xtr, ytr); preds["rf"] = m.predict(Xte); times["rf"] = time.time() - t
    # 4 MLP
    t = time.time(); m = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed); m.fit(Xtr, ytr); preds["mlp"] = m.predict(Xte); times["mlp"] = time.time() - t
    # 5 XGBoost
    t = time.time(); f, xm, present = fit_xgb(mk_xgb(seed), Xtr, ytr); preds["xgb"] = f(Xte); times["xgb"] = time.time() - t
    # 6 hybrid: MLP encoder (64-32) -> 32-d latent -> XGBoost
    t = time.time(); enc = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed); enc.fit(Xtr, ytr)
    def latent(X):
        A = X
        for i in range(len(enc.coefs_) - 1): A = np.maximum(A @ enc.coefs_[i] + enc.intercepts_[i], 0)
        return A
    fh, _, _ = fit_xgb(mk_xgb(seed), latent(Xtr), ytr); preds["hybrid"] = fh(latent(Xte)); times["hybrid"] = time.time() - t
    # 7 XGBoost without destination port (shortcut ablation)
    F2 = [c for c in F if c != "TCP Destination Port"]; Xtr2, Xte2, _ = prep(tr, te, F2)
    t = time.time(); f2, _, _ = fit_xgb(mk_xgb(seed), Xtr2, ytr); preds["xgb_noport"] = f2(Xte2); times["xgb_noport"] = time.time() - t
    R["models"] = {k: {**metrics(yte, p), "train_time_s": round(times[k], 2)} for k, p in preds.items()}
    R["seen_unseen"] = {k: {"acc_seen": float((preds[k][seen] == yte[seen]).mean()) if seen.any() else None,
                            "acc_unseen": float((preds[k][~seen] == yte[~seen]).mean()) if (~seen).any() else None}
                        for k in ["memoriser", "xgb", "hybrid", "rf"]}
    R["mcnemar_hybrid_vs_xgb"] = mcnemar(preds["hybrid"] == yte, preds["xgb"] == yte)
    R["mcnemar_rf_vs_xgb"] = mcnemar(preds["rf"] == yte, preds["xgb"] == yte)
    labs = np.unique(yte)
    R["classwise_xgb"] = {le.classes_[c]: v for c, v in zip(labs, f1_score(yte, preds["xgb"], labels=labs, average=None, zero_division=0).tolist())}
    R["classwise_recall_xgb"] = {le.classes_[c]: float((preds["xgb"][yte == c] == c).mean()) for c in labs}
    if seed == int(SEEDS.split(",")[0]):
        np.save(os.path.join(OUT, f"cm_xgb_{PROTO}.npy"), confusion_matrix(yte, preds["xgb"], labels=np.arange(K)))
        # SHAP on XGBoost (TreeExplainer), 2000 test rows
        import shap
        rs = np.random.RandomState(0); sidx = rs.choice(len(Xte), size=min(2000, len(Xte)), replace=False)
        sv = shap.TreeExplainer(xm).shap_values(Xte[sidx])
        sv = np.abs(np.array(sv)); 
        # shape handling: (n, feat, classes) or (classes, n, feat)
        if sv.ndim == 3 and sv.shape[1] == len(cols): imp = sv.mean(axis=(0, 2))
        elif sv.ndim == 3: imp = sv.mean(axis=(0, 1))
        else: imp = sv.mean(axis=0)
        R["shap_mean_abs"] = dict(zip(cols, map(float, imp)))
        if PROTO == "random":
            from boruta import BorutaPy
            np.int = int; np.float = float; np.bool = bool
            rs2 = np.random.RandomState(seed); bi = rs2.choice(len(Xtr), size=40000, replace=False)
            Xb, _, _ = prep(tr, te, F)
            bor = BorutaPy(RandomForestClassifier(n_jobs=2, max_depth=7, random_state=seed), n_estimators=100, max_iter=40, random_state=seed)
            bor.fit(Xb[bi][:, :8], ytr[bi])
            R["boruta"] = {c: {"confirmed": bool(s), "rank": int(r)} for c, s, r in zip(F, bor.support_, bor.ranking_)}
    R["classes"] = list(le.classes_); R["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(OUT, f"res_{PROTO}_s{seed}.json"), "w"), indent=1, default=float)
    print(PROTO, seed, {k: round(v["f1_macro"], 4) for k, v in R["models"].items()}, "seen", round(R["test_seen_frac"], 3), "t", R["elapsed_s"], flush=True)
