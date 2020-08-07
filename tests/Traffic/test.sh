for seed in 0 1 2 3 4
do
	# python dqn.py --seed $seed
	# python mydqn.py --seed $seed
	# python mydqn.py --seed $seed --cg 0.1
	python mydqn.py --seed $seed --cg 0.1 --expl 0.2
done
