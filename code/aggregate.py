import json, glob, numpy as np, sys
from scipy.stats import ttest_rel
R = {}
for f in sorted(glob.glob("results/res_*_s*.json")):
    d = json.load(open(f)); R.setdefault(d["protocol"], []).append(d)
S = {"protocols": {}}
for p, L in R.items():
    L = sorted(L, key=lambda d: d["seed"]); P = {"n_seeds": len(L), "seeds": [d["seed"] for d in L]}
    models = L[0]["models"].keys()
    P["models"] = {m: {k: [float(np.mean([d["models"][m][k] for d in L])), float(np.std([d["models"][m][k] for d in L], ddof=1)) if len(L) > 1 else 0.0]
                       for k in ["f1_macro", "bal_acc", "acc", "train_time_s"]} for m in models}
    P["test_seen_frac"] = [float(np.mean([d["test_seen_frac"] for d in L])), float(np.std([d["test_seen_frac"] for d in L], ddof=1)) if len(L) > 1 else 0]
    P["pool_test_seen_frac"] = [float(np.mean([d["pool_test_seen_frac"] for d in L])), float(np.std([d["pool_test_seen_frac"] for d in L], ddof=1)) if len(L) > 1 else 0]
    P["seen_unseen"] = {m: {k: float(np.nanmean([np.nan if d["seen_unseen"][m][k] is None else d["seen_unseen"][m][k] for d in L])) for k in ["acc_seen", "acc_unseen"]} for m in L[0]["seen_unseen"]}
    a = np.array([d["models"]["hybrid"]["f1_macro"] for d in L]); b = np.array([d["models"]["xgb"]["f1_macro"] for d in L])
    P["hybrid_minus_xgb_f1"] = [float(np.mean(a - b)), float(np.std(a - b, ddof=1)) if len(L) > 1 else 0.0]
    P["ttest_hybrid_xgb_p"] = float(ttest_rel(a, b).pvalue) if len(L) > 1 else None
    P["mcnemar_hybrid_xgb"] = [d["mcnemar_hybrid_vs_xgb"] for d in L]
    P["mcnemar_rf_xgb"] = [d["mcnemar_rf_vs_xgb"] for d in L]
    a2 = np.array([d["models"]["xgb_noport"]["f1_macro"] for d in L])
    P["noport_minus_xgb_f1"] = [float(np.mean(a2 - b)), float(np.std(a2 - b, ddof=1)) if len(L) > 1 else 0.0]
    cls = L[0]["classes"]
    P["classwise_xgb_f1"] = {c: float(np.mean([d["classwise_xgb"].get(c, np.nan) for d in L if c in d["classwise_xgb"]])) if any(c in d["classwise_xgb"] for d in L) else None for c in cls}
    P["classwise_xgb_recall"] = {c: float(np.mean([d["classwise_recall_xgb"][c] for d in L if c in d["classwise_recall_xgb"]])) if any(c in d["classwise_recall_xgb"] for d in L) else None for c in cls}
    P["n_test_classes"] = [d["models"]["xgb"]["n_test_classes"] for d in L]
    P["min_train_per_class"] = [int(min(d["train_per_class"])) for d in L]
    P["zero_train_classes"] = [[cls[i] for i, v in enumerate(d["train_per_class"]) if v == 0] for d in L]
    P["zero_test_classes"] = [[cls[i] for i, v in enumerate(d["test_per_class"]) if v == 0] for d in L]
    P["small_test_classes"] = [{cls[i]: v for i, v in enumerate(d["test_per_class"]) if v < 4500} for d in L]
    P["small_train_classes"] = [{cls[i]: v for i, v in enumerate(d["train_per_class"]) if v < 10500} for d in L]
    s0 = [d for d in L if "shap_mean_abs" in d]
    if s0: P["shap"] = s0[0]["shap_mean_abs"]
    b0 = [d for d in L if "boruta" in d]
    if b0: P["boruta"] = b0[0]["boruta"]
    S["protocols"][p] = P
json.dump(S, open("summary.json", "w"), indent=1)
for p, P in S["protocols"].items():
    print("==", p, P["n_seeds"], "seen", P["test_seen_frac"], "pool_seen", P["pool_test_seen_frac"])
    for m, v in P["models"].items(): print(f"  {m:11s} F1 {v['f1_macro'][0]:.4f}±{v['f1_macro'][1]:.4f}  BA {v['bal_acc'][0]:.4f}±{v['bal_acc'][1]:.4f} t {v['train_time_s'][0]:.1f}")
    print("  hyb-xgb", P["hybrid_minus_xgb_f1"], "t-p", P["ttest_hybrid_xgb_p"], "mcn p", [round(x["p"], 4) for x in P["mcnemar_hybrid_xgb"]])
    print("  noport-xgb", P["noport_minus_xgb_f1"], "seen/unseen", P["seen_unseen"])
    print("  ntestcls", P["n_test_classes"], "zero_train", P["zero_train_classes"], "zero_test", P["zero_test_classes"])
