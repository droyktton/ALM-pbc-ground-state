#usage
#python plot_S.py S_avg_L=8192_ANH=2.txt --out S_avg.png
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="S_avg file (two columns: k, S(k))")
parser.add_argument("--out", default="S_avg.png", help="output PNG filename")
args = parser.parse_args()

data = np.loadtxt(args.input)
k = data[:, 0]
S = data[:, 1]

# remove k=0 to avoid log issues
mask = k > 0
k = k[mask]
S = S[mask]

plt.figure()
plt.loglog(k, S)
plt.xlabel("k")
plt.ylabel("S(k)")
plt.tight_layout()
plt.savefig(args.out, dpi=300)
plt.close()
