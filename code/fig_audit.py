import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
A=json.load(open("audit_full.json"))
cr=A["ceiling_recall_by_class"]; cf=A["rows_in_conflict_fp_pct_by_class"]
cls=sorted(cr,key=lambda c:cr[c])
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":6.5,"axes.edgecolor":"#c3c2b7","axes.linewidth":0.6})
fig,ax=plt.subplots(figsize=(3.3,3.5),dpi=300)
y=np.arange(len(cls)); h=0.38
b1=ax.barh(y+h/2,[cr[c] for c in cls],height=h-0.04,color="#2a78d6",label="Ceiling recall (max achievable)")
b2=ax.barh(y-h/2,[cf[c] for c in cls],height=h-0.04,color="#eb6834",hatch="////",edgecolor="white",lw=0,label="Share of packets in shared fingerprints")
ax.set_yticks(y); ax.set_yticklabels([c.replace("_"," ") for c in cls])
ax.set_xlim(0,1.0); ax.set_xticks([0,0.25,0.5,0.75,1.0])
ax.xaxis.grid(True,color="#e1e0d9",lw=0.5); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors="#52514e",length=2)
for i,c in enumerate(cls):
    if cr[c]<0.7: ax.text(cr[c]+0.01,i+h/2,f"{cr[c]:.2f}",va="center",fontsize=5.6,color="#0b0b0b")
ax.set_xlabel("Proportion of class packets",color="#52514e")
ax.legend(loc="lower center",bbox_to_anchor=(0.42,1.0),ncol=1,frameon=False,fontsize=6)
fig.tight_layout(); fig.savefig("fig/fig2_audit.png",dpi=300); print(cls)
