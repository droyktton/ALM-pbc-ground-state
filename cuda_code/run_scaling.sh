#!/bin/bash

####################################################
# Parameters
####################################################

NSAMPLES=20

FITMIN=2
FITMAX=20

LSIZES=(
1024
2048
4096
8192
16384
32768
65536
131072
262144
524288
1048576
#2097152
#4194304
#8388608
#16777216
)

####################################################
# Output file
####################################################

echo "# L zeta dzeta R2" > zeta_vs_L.dat

####################################################
# Loop over system sizes
####################################################

for L in "${LSIZES[@]}"
do

    DIR=L${L}

    mkdir -p "$DIR"

    echo
    echo "==========================================="
    echo "L = $L"
    echo "==========================================="

    ################################################
    # Generate configurations
    ################################################

    for ((seed=1; seed<=NSAMPLES; seed++))
    do

        FILE=$(printf "%s/u_%08d.bin" "$DIR" "$seed")

        if [ -f "$FILE" ]; then
            continue
        fi

        ./solver "$L" "$seed" "$DIR"

    done

    ################################################
    # Analyze
    ################################################

    (
        cd "$DIR"
        python3 ../structure_factor.py "$FITMIN" "$FITMAX"
    )

    ################################################
    # Collect fit results
    ################################################

    cat "${DIR}/fit.dat" >> zeta_vs_L.dat

done

####################################################
# Plot zeta(L)
####################################################

python3 << EOF

import numpy as np
import matplotlib.pyplot as plt

L,z,e,r = np.loadtxt("zeta_vs_L.dat", unpack=True)

plt.figure(figsize=(6,4))

plt.errorbar(
    L,
    z,
    yerr=e,
    fmt='o-',
    capsize=3
)

plt.xscale("log")

plt.xlabel(r"$L$")
plt.ylabel(r"$\zeta$")

plt.grid(True)

plt.tight_layout()

plt.savefig("zeta_vs_L.png", dpi=300)

plt.show()

EOF

echo
echo "==========================================="
echo "Finished."
echo "==========================================="
echo
echo "Summary : zeta_vs_L.dat"
echo "Figure  : zeta_vs_L.png"