for seed in 0
do
	for exp_name in max2
	do
		python prg_gaussian.py --oa --re --ce --exp_name $exp_name --seed $seed
		python maddpg.py --oa --re --exp_name $exp_name --seed $seed
		python masac.py --oa --re --exp_name $exp_name --seed $seed
		# python prg_gaussian.py --ce --exp_name $exp_name --seed $seed
		# python maddpg.py --exp_name $exp_name --seed $seed
		# python masac.py --exp_name $exp_name --seed $seed
		# python prg_gaussian.py --ta --ce --exp_name $exp_name --seed $seed
	done
done