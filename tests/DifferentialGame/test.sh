for seed in 0 1 2 3 4
do
	# python maddpg.py --epoch 200 --exp_name zero_sum --seed $seed
	# python maddpg.py --online_action --epoch 200 --exp_name zero_sum --seed $seed
	# python masac.py --epoch 200 --exp_name zero_sum --seed $seed
	# python masac.py --online_action --epoch 200 --exp_name zero_sum --seed $seed
	# python prg.py --epoch 200 --k 1 --exp_name zero_sum --seed $seed
	python prg.py --epoch 200 --k 2 --exp_name zero_sum --seed $seed
done