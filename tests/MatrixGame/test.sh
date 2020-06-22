for exp_name in zero_sum
do
	for seed in 0 1 2
	do
		python masac_discrete.py --online_action --exp_name $exp_name --seed $seed
		python prg_discrete.py --online_action --use_gumbel --exp_name $exp_name --seed $seed
	done
done