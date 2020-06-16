for seed in 0 1 2 3 4
# for seed in 5 6 7 8 9
do
	# python prg.py --online_action --exp_name zero_sum --seed $seed
	# python prg_gaussian.py --online_action --centropy --exp_name zero_sum --seed $seed
	# python prg_gaussian.py --online_action --exp_name zero_sum --seed $seed
	python prg_gaussian.py --online_action --exp_name cooperative --seed $seed
done