#!/bin/bash
# Example: sweep over several n values, extract zeta_s and W^2(L),
# and collect results into a CSV.

OUTFILE="scan_results.csv"
echo "L,n,c,delta,samples,zeta_s,zeta_s_err,W2_direct,W2_parseval" > "$OUTFILE"

for n in 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0 3.0 4.0 5.0 10.0; do
    for L in 1024 2048 4096 8192; do
    
        nsamples=$(echo "1024*1024/$L" | bc)  # Number of samples to average over for each (L,n) pair
        
        # Run the tool, capture full stdout, skip plotting/showing to keep it fast
        output=$(python3 run_structure_factor.py -L "$L" -n "$n" --samples $nsamples --qmin 0.001 --qmax 0.1 -c 0.0 --no-plot)
    
        # Grab the single machine-readable line and turn it into a CSV row.
        # The RESULT line looks like:
        #   RESULT L=4096 n=2.0 c=1.0 delta=1.0 samples=200 zeta_s=1.4164 ...
        result_line=$(echo "$output" | grep "^RESULT")
    
        zeta_s=$(echo "$result_line"   | grep -oP 'zeta_s=\K[0-9.eE+-]+')
        zeta_s_err=$(echo "$result_line" | grep -oP 'zeta_s_err=\K[0-9.eE+-]+')
        W2_direct=$(echo "$result_line"  | grep -oP 'W2_direct=\K[0-9.eE+-]+')
        W2_parseval=$(echo "$result_line" | grep -oP 'W2_parseval=\K[0-9.eE+-]+')
    
        echo "$L,$n,1.0,1.0,$nsamples,$zeta_s,$zeta_s_err,$W2_direct,$W2_parseval" >> "$OUTFILE"
    
        echo "L=$L samples=$nsamples n=$n -> zeta_s=$zeta_s +/- $zeta_s_err   W2=$W2_direct"
    done
done

echo "Saved sweep results to $OUTFILE"
