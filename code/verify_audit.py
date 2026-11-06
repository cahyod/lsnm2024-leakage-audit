"""Independent re-computation of R, C and BA* with numpy row-unique (no hashing)."""
import numpy as np, pandas as pd, json, sys
F=["Length","IP Length","IP TTL","TCP Length","TCP Window Size","UDP Length","ICMP Type","TCP Destination Port"]
df=pd.read_pickle(sys.argv[1])
X=np.nan_to_num(df[F].to_numpy(dtype=np.float64),nan=-1.0)
y=df["cls"].astype(str).to_numpy()
_,inv=np.unique(X,axis=0,return_inverse=True); inv=inv.ravel()
N=len(X); U=inv.max()+1
classes,yi=np.unique(y,return_inverse=True); K=len(classes)
M=np.zeros((U,K),dtype=np.int64); np.add.at(M,(inv,yi),1)
R=1-U/N
labels_per=(M>0).sum(1); C=(labels_per[inv]>1).mean()
Nk=M.sum(0); BA=(M/Nk).max(1).sum()/K
print(json.dumps({"N":int(N),"U":int(U),"R":round(R,6),"C":round(float(C),6),"BA*":round(float(BA),6),"conflict_fp":int((labels_per>1).sum())}))
