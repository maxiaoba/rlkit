for seed in 0 1 2
do
	# python maddpg.py --exp_name simple_adversary --epoch 1000 --seed $seed --gpu
	# python masac.py --exp_name simple_adversary --epoch 1000 --seed $seed --gpu
	python prg_gaussian.py --exp_name simple_adversary --epoch 1000 --online_action --centropy --seed $seed --gpu
done
