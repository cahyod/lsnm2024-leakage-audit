import json, glob, numpy as np
from scipy.stats import ttest_rel, t as tdist, spearmanr
def ci(x):
    x=np.asarray(x); n=len(x); m=x.mean(); h=tdist.ppf(0.975,n-1)*x.std(ddof=1)/np.sqrt(n); return [float(m),float(m-h),float(m+h)]
OUT={}
for p in ["random","temporal","unseen"]:
    C={json.load(open(f))["seed"]:json.load(open(f)) for f in glob.glob(f"results/cf_{p}_s*.json")}
    R={json.load(open(f))["seed"]:json.load(open(f)) for f in glob.glob(f"results/res_{p}_s*.json")}
    seeds=sorted(set(C)&set(R)); 
    if not seeds: continue
    x=np.array([R[s]["models"]["xgb"]["f1_macro"] for s in seeds]); xc=np.array([C[s]["models"]["xgb"]["f1_macro"] for s in seeds])
    hc=np.array([C[s]["models"]["hybrid_cf"]["f1_macro"] for s in seeds]); hi=np.array([R[s]["models"]["hybrid"]["f1_macro"] for s in seeds])
    d={"n":len(seeds),"xgb_reproduced_maxabsdiff":float(np.abs(x-xc).max())}
    for nm,h in [("insample",hi),("crossfit",hc)]:
        tt=ttest_rel(h,x); d[nm]={"f1":[float(h.mean()),float(h.std(ddof=1))],"delta_ci":ci(h-x),"t":float(tt.statistic),"p":float(tt.pvalue)}
    mc=[C[s]["mcnemar_hybridcf_vs_xgb"] for s in seeds]
    d["crossfit"]["mcn_h"]=sum(1 for m in mc if m["p"]<0.05 and m["a_only"]>m["b_only"]); d["crossfit"]["mcn_x"]=sum(1 for m in mc if m["p"]<0.05 and m["a_only"]<m["b_only"])
    mi=[R[s]["mcnemar_hybrid_vs_xgb"] for s in seeds]
    d["insample"]["mcn_h"]=sum(1 for m in mi if m["p"]<0.05 and m["a_only"]>m["b_only"]); d["insample"]["mcn_x"]=sum(1 for m in mi if m["p"]<0.05 and m["a_only"]<m["b_only"])
    d["crossfit"]["time_ratio"]=float(np.mean([C[s]["models"]["hybrid_cf"]["train_time_s"]/C[s]["models"]["xgb"]["train_time_s"] for s in seeds]))
    d["insample"]["time_ratio"]=float(np.mean([R[s]["models"]["hybrid"]["train_time_s"]/R[s]["models"]["xgb"]["train_time_s"] for s in seeds]))
    su=lambda k,m: float(np.nanmean([np.nan if C[s]["seen_unseen"][m][k] is None else C[s]["seen_unseen"][m][k] for s in seeds]))
    d["crossfit"]["seen_unseen"]={m:{"seen":su("acc_seen",m),"unseen":su("acc_unseen",m)} for m in ["xgb","hybrid_cf"]}
    sh=[C[s]["shap_mean_abs"] for s in seeds]; keys=sorted(sh[0])
    shares=np.array([[v[k]/sum(v.values()) for k in keys] for v in sh])
    d["shap_share_mean"]=dict(zip(keys,shares.mean(0).tolist())); d["shap_share_sd"]=dict(zip(keys,shares.std(0,ddof=1).tolist()))
    d["shap_within_rho"]=float(np.mean([spearmanr(shares[i],shares[j]).correlation for i in range(len(sh)) for j in range(i+1,len(sh))]))
    d["xgb_f1_ci"]=ci(x)
    OUT[p]=d
keys=sorted(OUT[next(iter(OUT))]["shap_share_mean"])
for a,b in [("random","temporal"),("random","unseen")]:
    if a in OUT and b in OUT:
        OUT[f"rho_{a}_{b}"]=float(spearmanr([OUT[a]["shap_share_mean"][k] for k in keys],[OUT[b]["shap_share_mean"][k] for k in keys]).correlation)
json.dump(OUT,open("results/summary_cf.json","w"),indent=1)
for p,d in OUT.items():
    if isinstance(d,dict): print(p,json.dumps({k:d[k] for k in ["n","xgb_reproduced_maxabsdiff","insample","crossfit","xgb_f1_ci"]},indent=0)[:900])
    else: print(p,d)
