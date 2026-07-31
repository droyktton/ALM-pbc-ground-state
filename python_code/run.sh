for L in 31768 65536 131072 262144 524288
do 
	echo "L="$L
	for n in 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 3.0 4.0 5.0 6.0
	do 
		zetas=$(python3 run_structure_factor.py -L $L -n $n -c 0.0 --samples 20 --no-show --qfrac 0.15 | grep "spectral"); 
		echo $n $zetas; 
	done > L$L; 
done


gnuplot -e \
"plot [:1.6][:1.55] for[L in \"65536 131072 4194304\"] sprintf('L%s',L) u (1./(2*\$1-1.0)):(\$6):8 w errorl t \"L=\".L,1.5, \
1+x/2+0.25 t '1+p/2+1/4' "
