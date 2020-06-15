for seed in 0 1 2 3 4
do
	# python prg.py --online_action --exp_name zero_sum --seed $seed
	python prg_gaussian.py --online_action --centropy --exp_name zero_sum --seed $seed
done