import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":6.0})
W,H=3.3,4.6
fig=plt.figure(figsize=(W,H),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,140); ax.axis("off")
def box(x,y,w,h,txt,fc="#FFFFFF",ec="#333333",bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.5",fc=fc,ec=ec,lw=0.7))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontweight="bold" if bold else "normal",linespacing=1.25)
def arrow(x1,y1,x2,y2):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.7,color="#333333",mutation_scale=7))
box(8,124,84,13,"LSNM2024 Wireshark exports: 21 CSV captures,\n4,543,916 packets, 16 classes\n(label inherited from capture folder)",fc="#EEF3FA",bold=False)
arrow(50,124,50,119)
box(8,104,84,15,"Eight header fields per packet: Length, IP Length, IP TTL,\nTCP Length, TCP Window, UDP Length, ICMP Type,\nTCP Dst Port; packet fingerprint "+r"$\phi(x)$",fc="#FFFFFF")
arrow(30,104,22,99); arrow(70,104,78,99)
box(1,78,42,21,"Corpus audit\nredundancy R, Eq. (1)\nconflict share C, Eq. (2)\nceiling "+r"$BA^{*}$"+", Eq. (3)",fc="#FFF6E6")
box(51,78,48,21,"Split protocols (5 repetitions)\nP1 random packet split\nP2 temporal 70/30 per capture\nP3 unseen-fingerprint split",fc="#FFF6E6")
arrow(75,78,75,72)
box(51,56,48,16,"Class-balanced sampling\n10,500 train / 4,500 test\nper class; median imputation\n+ 4 protocol-presence flags",fc="#FFFFFF")
arrow(75,56,75,53)
box(8,34,91,19,"Models (fixed hyperparameters): fingerprint memoriser,\nlogistic regression, random forest, MLP, XGBoost,\nhybrid MLP-encoder "+r"$\rightarrow$"+" XGBoost, XGBoost without\ndestination port; Boruta relevance screening",fc="#EEF7EE")
arrow(53.5,34,53.5,28)
box(8,9,91,19,"Evaluation: macro-F1, balanced accuracy,\naccuracy on seen vs unseen fingerprints,\nexact McNemar test (Eq. 4), training time,\nTreeSHAP attribution per protocol",fc="#EEF7EE")
arrow(22,78,22,53.5); ax.text(23,64,"bounds",rotation=90,va="center",fontsize=5.8,color="#555555")
fig.savefig("fig/fig1_flow.png",dpi=300); print("ok")
