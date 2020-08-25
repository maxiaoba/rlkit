for seed in 0 1 2
do
	for obs in 1 3
	do
	python test_sup_online.py --seed $seed --obs $obs
	python ppo_sup_online.py --seed $seed --sw 1.0 --obs $obs
	python ppo.py --seed $seed --obs $obs
	done
done
