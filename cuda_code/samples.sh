for a in 2 4 6 8 10 12 14 16 18 20
do
	for L in 4096 #512 #1024 #2048 4096 8192
	do
		make clean
		make SIZEL=$L ANHN=$a

		for((s=42;s<52;s++))
		do			
			dir="L"$L"_ANH"$a"_seed"$s			

			nom="L="$L"ANH="$a"seed="$s			

			echo $dir

			mkdir $dir
			cd $dir
				#cp ../jobGPU .
				cp ../job_slurm.sh .
				cp ../alm .
				#qsub -N $nom jobGPU $s
				sbatch --job-name=$nom job_slurm.sh $s
			cd ../
		done
	done
done
