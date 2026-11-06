## Companion manuscript
This repository accompanies "Leakage-aware evaluation of packet-level intrusion detection on LSNM2024" (submitted to Jurnal RESTI). Push this repository to your own remote and put the resulting URL in Section 2.4 of the manuscript in place of the placeholder.

# Supplementary material: leakage-aware evaluation of packet-level intrusion detection on LSNM2024

## Data
LSNM2024 version 1 (Mendeley Data, doi:10.17632/7pzyfvv9jn.1). Unzip `LSNM2024 Dataset.zip` and point the scripts to the
`Dataset-Ready (Use This)` folder. Only the 21 CSV exports are used; the two PCAP files are ignored.

## Pipeline (Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1, BorutaPy 0.4.3)
1. `python load_full.py "<.../Dataset-Ready (Use This)>" lsnm_full.pkl`
   Parses every CSV (4,543,916 packets, 16 classes; label = capture folder).
2. `python audit_full.py lsnm_full.pkl audit_full.json`
   Model-free audit: redundancy R (Eq. 1), conflict share C (Eq. 2), class-balanced ceiling BA* (Eq. 3),
   per-class ceiling recall, single-field ceilings, protocol composition. Also writes `audit_full_fp.pkl` (fingerprints).
3. `python run_exp.py lsnm_full.pkl audit_full_fp.pkl results <random|temporal|unseen> 0,1,2,3,4`
   Split protocols P1/P2/P3, class-balanced sampling (10,500 train / 4,500 test per class), seven models,
   seen/unseen accuracy, exact McNemar tests, TreeSHAP (first seed; multi-seed SHAP is in step 7) and Boruta (P1, first seed).
4. `python aggregate.py` -> `summary.json` (means, SDs, paired tests).
5. `fig_flow.py`, `fig_audit.py`, `fig_results.py` regenerate the figures.

Hyperparameters are fixed in `run_exp.py` and were not tuned. Seeds 0-4 reproduce the reported repetitions.
`results/` holds one JSON per protocol and seed, plus the XGBoost confusion matrix of the first seed per protocol.
6. `python verify_audit.py lsnm_full.pkl` independently recomputes R, C and BA* without hashing (matches audit_full.json).
7. `python run_cf.py lsnm_full.pkl audit_full_fp.pkl results <random|temporal|unseen> 0,1,2,3,4`
   Cross-fitted hybrid (2-fold: encoder fitted on one half, XGBoost on the encoded other half, swapped, probabilities averaged),
   timing, seen/unseen accuracy and TreeSHAP shares for every seed. Writes `cf_<protocol>_s<seed>.json`.
8. `python aggregate_cf.py` -> `summary_cf.json` (paired differences with 95% t-based CIs, paired t (df), McNemar counts,
   time ratios, multi-seed SHAP means/SDs and Spearman correlations between protocols). Table 3 and Figure 3 use this file.

All paths in the scripts are relative to the folder that contains `code/`, `results/` and `figures/`.
