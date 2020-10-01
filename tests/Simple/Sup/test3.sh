for seed in 0 1 2
do
	for obs in 5
	do
		python ppo.py --hidden 24 --seed $seed --obs $obs	
		python ppo_sup_vanilla.py --hidden 24 --seed $seed --obs $obs
		python ppo_sup.py --hidden 24 --seed $seed --obs $obs
		python ppo_sup_online.py --hidden 24 --seed $seed --obs $obs
		python ppo_sup_sep2.py --hidden 16 --seed $seed --obs $obs
		python test_sup.py --hidden 24 --seed $seed --obs $obs
	done
done
