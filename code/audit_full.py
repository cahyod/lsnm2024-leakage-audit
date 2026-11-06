import pandas as pd, numpy as np, json, sys
full=pd.read_pickle(sys.argv[1]); out=sys.argv[2]
F=["Length","IP Length","IP TTL","TCP Length","TCP Window Size","UDP Length","ICMP Type","TCP Destination Port"]
res={}
cnt=full['cls'].value_counts(); res['class_counts']=cnt.to_dict(); print(cnt)
# label column values
for c,g in full.groupby('cls',observed=True): print('label col',c,g['label'].astype(str).value_counts(dropna=False).head(4).to_dict())
# protocol composition
HK={'ARP','DNS','MDNS','NBNS','LLMNR','SSDP','NTP','BROWSER','DHCP','DHCPv6','IGMPv2','IGMPv3','ICMPv6','LLC','STP','CDP','LLDP','NBSS','SMB','IGMP'}
comp={}
for c,g in full.groupby('cls',observed=True):
    vc=g['Protocol'].value_counts(normalize=True)
    hk=set(HK)-({'ARP'} if c=='ARP_Spoofing' else set())
    comp[c]={'top':{k:round(v,4) for k,v in vc.head(5).items()},'housekeeping_share':round(float(g['Protocol'].isin(hk).mean()),4)}
    print(c, comp[c])
res['protocol_composition']=comp
# top source IPs per class
srcs={}
for c,g in full.groupby('cls',observed=True):
    srcs[c]={str(k):round(v,3) for k,v in g['IP Source'].value_counts(normalize=True,dropna=False).head(4).items()}
print(json.dumps(srcs,indent=0)[:3000])
res['top_ip_sources']=srcs
# fingerprints on 8 features (NaN kept as distinct sentinel)
X=full[F].fillna(-1.0)
fp=pd.util.hash_pandas_object(X,index=False).values
full['fp']=fp
nuniq=pd.Series(fp).nunique()
dup_keep_false=pd.Series(fp).duplicated(keep=False).mean()
res['n_rows']=len(full); res['n_unique_fp']=int(nuniq); res['dup_rows_keepfalse_pct']=round(100*dup_keep_false,2)
res['redundant_rows_pct']=round(100*(1-nuniq/len(full)),2)
# per-class unique fps
pcu=full.groupby('cls',observed=True)['fp'].nunique(); res['unique_fp_per_class']=pcu.to_dict(); print(pcu)
# label conflicts
ct=full.groupby(['fp','cls'],observed=True).size().rename('n').reset_index()
nl=ct.groupby('fp')['cls'].nunique()
conf_fp=set(nl[nl>1].index)
res['n_conflict_fp']=int(len(conf_fp))
res['rows_in_conflict_fp_pct']=round(100*full['fp'].isin(conf_fp).mean(),2)
res['rows_in_conflict_fp_pct_by_class']=full.assign(c=full['fp'].isin(conf_fp)).groupby('cls',observed=True)['c'].mean().round(4).to_dict()
# class-balanced Bayes ceiling (max macro recall / balanced accuracy for any deterministic f(x))
w=1.0/cnt
ct['wn']=ct['n']*ct['cls'].map(w).astype(float)
best=ct.loc[ct.groupby('fp')['wn'].idxmax()]
res['balanced_acc_ceiling']=round(float(best['wn'].sum()/len(cnt)),4)
# per class ceiling recall
pcr=(best.groupby('cls',observed=True)['n'].sum()/cnt).reindex(cnt.index).fillna(0)
res['ceiling_recall_by_class']=pcr.round(4).to_dict()
# prior-weighted (natural) ceiling accuracy
best2=ct.loc[ct.groupby('fp')['n'].idxmax()]
res['natural_acc_ceiling']=round(float(best2['n'].sum()/len(full)),4)
print({k:res[k] for k in ['n_rows','n_unique_fp','dup_rows_keepfalse_pct','redundant_rows_pct','n_conflict_fp','rows_in_conflict_fp_pct','balanced_acc_ceiling','natural_acc_ceiling']})
print(res['ceiling_recall_by_class'])
# destination-port-only and TTL-only balanced ceilings (shortcut check)
for col in ['TCP Destination Port','IP TTL','Length']:
    k=full[col].fillna(-1.0)
    t=full.groupby([k,'cls'],observed=True).size().rename('n').reset_index()
    t['wn']=t['n']*t['cls'].map(w).astype(float)
    b=t.loc[t.groupby(col)['wn'].idxmax()]
    res[f'balanced_ceiling_only_{col}']=round(float(b['wn'].sum()/len(cnt)),4)
    print(col,res[f'balanced_ceiling_only_{col}'])
missing=(full[F].isna().mean()*100).round(2).to_dict(); res['missing_pct']=missing; print(missing)
full[['fp']].to_pickle(out.replace('.json','_fp.pkl'))
json.dump(res,open(out,'w'),indent=1,default=str)
