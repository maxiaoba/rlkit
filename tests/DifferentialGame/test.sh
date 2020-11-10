for seed in 0
do
	for exp_name in max2
	do
		python prg_gaussian.py --oa --ce --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python maddpg.py --oa --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python masac.py --oa --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python prg_gaussian.py --oa --re --ce --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python maddpg.py --oa --re --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
		python masac.py --oa --re --exp_name $exp_name --seed $seed --hidden 32 --epoch 1000
	done
done