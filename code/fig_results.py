import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
S=json.load(open("results/summary.json"))["protocols"]; A=json.load(open("results/audit_full.json"))
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":6.5,"axes.edgecolor":"#c3c2b7","axes.linewidth":0.6})
COL={"random":"#2a78d6","temporal":"#eb6834","unseen":"#1baf7a"}; HAT={"random":"","temporal":"////","unseen":"...."}
LAB={"random":"P1 random","temporal":"P2 temporal","unseen":"P3 unseen fingerprint"}
PR=[p for p in ["random","temporal","unseen"] if p in S]
# ---- Fig 3: balanced accuracy by model and protocol vs ceiling ----
models=["memoriser","logreg","mlp","rf","xgb","hybrid","xgb_noport"]
names={"memoriser":"Memoriser","logreg":"Logistic reg.","mlp":"MLP","rf":"Random forest","xgb":"XGBoost","hybrid":"Encoder+XGBoost","xgb_noport":"XGBoost, no port"}
fig,ax=plt.subplots(figsize=(3.3,3.4),dpi=300)
y=np.arange(len(models))[::-1]; h=0.26
for j,p in enumerate(PR):
    m=[S[p]["models"][k]["bal_acc"][0] for k in models]; e=[S[p]["models"][k]["bal_acc"][1] for k in models]
    ax.barh(y+(1-j)*h,m,height=h-0.03,xerr=e,color=COL[p],hatch=HAT[p],edgecolor="white",lw=0,label=LAB[p],
            error_kw=dict(lw=0.5,capsize=1.2,capthick=0.5,ecolor="#52514e"))
ax.axvline(A["balanced_acc_ceiling"],color="#0b0b0b",ls="--",lw=0.7)
ax.text(A["balanced_acc_ceiling"]+0.005,y[-1]-0.62,f"BA* = {A['balanced_acc_ceiling']:.3f}",fontsize=5.8,va="center")
ax.set_yticks(y); ax.set_yticklabels([names[k] for k in models]); ax.set_xlim(0,1.0)
ax.xaxis.grid(True,color="#e1e0d9",lw=0.5); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors="#52514e",length=2); ax.set_xlabel("Balanced accuracy (mean ± SD, 5 repetitions)",color="#52514e")
ax.legend(loc="lower center",bbox_to_anchor=(0.45,1.0),ncol=3,frameon=False,fontsize=5.8,handlelength=1.6,columnspacing=0.8)
ax.set_ylim(y[-1]-0.8,y[0]+0.5)
fig.tight_layout(); fig.savefig("figures/fig3_protocols.png",dpi=300)
# ---- Fig 4: SHAP share by input and protocol ----
inputs=["TCP Destination Port","TCP Window Size","Length","TCP Length","IP Length","IP TTL","ICMP Type","UDP Length","has_icmp","has_udp","has_tcp","has_ip"]
lab={"TCP Destination Port":"TCP dst port","TCP Window Size":"TCP window","Length":"Frame length","TCP Length":"TCP length","IP Length":"IP length","IP TTL":"IP TTL","ICMP Type":"ICMP type","UDP Length":"UDP length","has_icmp":"ICMP present","has_udp":"UDP present","has_tcp":"TCP present","has_ip":"IP present"}
fig,ax=plt.subplots(figsize=(3.3,2.75),dpi=300)
y=np.arange(len(inputs))[::-1]
for j,p in enumerate(PR):
    SC=json.load(open("results/summary_cf.json")); m=SC[p]["shap_share_mean"]; e=SC[p]["shap_share_sd"]
    ax.barh(y+(1-j)*h,[m[k] for k in inputs],height=h-0.03,xerr=[e[k] for k in inputs],color=COL[p],hatch=HAT[p],edgecolor="white",lw=0,label=LAB[p],error_kw=dict(lw=0.5,capsize=1.0,capthick=0.5,ecolor="#52514e"))
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in inputs])
ax.xaxis.grid(True,color="#e1e0d9",lw=0.5); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors="#52514e",length=2); ax.set_xlabel("Share of total mean |SHAP| (XGBoost, mean \u00b1 SD)",color="#52514e")
ax.legend(loc="lower center",bbox_to_anchor=(0.45,1.0),ncol=3,frameon=False,fontsize=5.8,handlelength=1.6,columnspacing=0.8)
ax.set_ylim(y[-1]-0.6,y[0]+0.5)
fig.tight_layout(); fig.savefig("figures/fig4_shap.png",dpi=300)
print("figs ok")
