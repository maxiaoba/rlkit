for seed in 1 2
do
	for obs in 1
	do
		python ppo_sup_lstm.py --hidden 16 --seed $seed --obs $obs
	done
done
