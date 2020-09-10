for seed in 0 1 2
do
	for obs in 1 3 5 10
	do
		python ppo_sup_online.py --sw 0.1 --seed $seed --obs $obs
	done
done
