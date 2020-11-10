for seed in 0
do
	for exp_name in zero_sum cooperative max2
	do
		python prg_gaussian.py --oa --ce --exp_name $exp_name --seed $seed
		python maddpg.py --oa --exp_name $exp_name --seed $seed --epoch 5
		python masac.py --oa --exp_name $exp_name --seed $seed --epoch 5
		python prg_gaussian.py --ce --exp_name $exp_name --seed $seed
		python maddpg.py --exp_name $exp_name --seed $seed --epoch 5
		python masac.py --exp_name $exp_name --seed $seed --epoch 5
		python prg_gaussian.py --ta --ce --exp_name $exp_name --seed $seed
	done
done