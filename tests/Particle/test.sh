for seed in 0 1 2
do
	# python masac.py --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python masac.py --online_action --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python maddpg.py --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python maddpg.py --online_action --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python prg_gaussian.py --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python prg_gaussian.py --online_action --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python prg.py --exp_name simple_spread --epoch 1000 --seed $seed --gpu
	# python prg.py --online_action --exp_name simple_spread --epoch 1000 --seed $seed --gpu
done
