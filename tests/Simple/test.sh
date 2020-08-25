for seed in 0 1 2
do
	for obs in 5
	do
		python trpo.py --seed $seed --obs $obs
		python trpo_sup.py --seed $seed --sw 0.1 --obs $obs
		python trpo_sup2.py --seed $seed --sw 0.1 --obs $obs
		python test_sup.py --seed $seed --obs $obs
	done
done
