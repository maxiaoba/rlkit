for seed in 0 1 2
do
	for obs in 1 3 5 10
	do
		python ppo_sup_sep2.py --seed $seed --obs $obs
	done
done
