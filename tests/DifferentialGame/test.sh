for seed in 0 1 2
do
	for exp_name in max2
	do
		python prg_gaussian.py --oa --ce --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python prg_mixgaussian.py --oa --ce --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python masac_gaussian.py --oa --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python masac_mixgaussian.py --oa --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python maddpg.py --oa --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
	done
done