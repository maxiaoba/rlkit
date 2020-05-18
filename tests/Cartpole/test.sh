for seed in 1 2 3 4
do
	# python maddpg_discrete.py --seed $seed
	# python maddpg_discrete.py --soft --seed $seed
	python prg_discrete.py --seed $seed
	python prg_discrete.py --soft --seed $seed
done