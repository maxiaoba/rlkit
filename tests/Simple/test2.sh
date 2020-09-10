for seed in 0 1 2
do
	for obs in 1 3 5 10
	do
		python test_sup.py --seed $seed --obs $obs
	done
done
