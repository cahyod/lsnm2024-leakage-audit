"""Parse every LSNM2024 'Dataset-Ready' CSV into one compact pickle (packet level)."""
import os, glob, sys, time
import numpy as np, pandas as pd
BASE = sys.argv[1]; OUT = sys.argv[2]
CLASS_MAP = {"Benign":"Benign","Malicious/DDOS RAW":"DDoS_RAW","Malicious/syn flood":"SYN_Flood",
 "Malicious/FTP Brute Force":"FTP_BruteForce","Malicious/ARP-spoof":"ARP_Spoofing",
 "Malicious/Port Scanning":"Port_Scanning","Malicious/SSH Brute Force":"SSH_BruteForce",
 "Malicious/Remote Code Execution":"RCE","Malicious/ICMP Flood":"ICMP_Flood","Malicious/Fuzzing":"Fuzzing",
 "Malicious/DDOS icmp":"DDoS_ICMP","Malicious/DOS":"DOS","Malicious/XSS":"XSS",
 "Malicious/Exploiting FTP":"FTP_Exploit","Malicious/SQL injection":"SQL_Injection","Malicious/DDOS-UDP":"DDoS_UDP"}
FEATS = ["Length","IP Length","IP TTL","TCP Length","TCP Window Size","UDP Length","ICMP Type","TCP Destination Port"]
META_NUM = ["Frame Time (Epoch)","TCP Stream","TCP Source Port","UDP Source Port","UDP Destination Port"]
META_STR = ["Protocol","Ethernet Source","Ethernet Destination","IP Source","IP Destination","label"]
frames=[]; t0=time.time(); fid=0
for rel,lab in CLASS_MAP.items():
    for p in sorted(glob.glob(os.path.join(BASE,rel,"*.csv"))):
        hdr = pd.read_csv(p,nrows=0,encoding_errors="replace").columns.tolist()
        use = [c for c in FEATS+META_NUM+META_STR if c in hdr]
        df = pd.read_csv(p,usecols=use,encoding_errors="replace",low_memory=False,on_bad_lines="skip",dtype=str)
        for c in FEATS+META_NUM:
            df[c] = pd.to_numeric(df[c],errors="coerce") if c in df else np.nan
        for c in META_STR:
            if c not in df: df[c]=None
        df["cls"]=lab; df["file_id"]=fid; df["file"]=os.path.relpath(p,BASE); df["row_in_file"]=np.arange(len(df))
        frames.append(df[FEATS+META_NUM+META_STR+["cls","file_id","file","row_in_file"]])
        print(f"{lab:16s} {os.path.relpath(p,BASE):55s} rows={len(df):>9d} t={time.time()-t0:.0f}s",flush=True)
        fid+=1
full=pd.concat(frames,ignore_index=True)
for c in ["Protocol","Ethernet Source","Ethernet Destination","IP Source","IP Destination","label","cls","file"]:
    full[c]=full[c].astype("category")
for c in FEATS+META_NUM: full[c]=full[c].astype("float64")
full.to_pickle(OUT)
print("TOTAL",len(full),"saved",OUT,"mem MB",full.memory_usage(deep=True).sum()/1e6)
