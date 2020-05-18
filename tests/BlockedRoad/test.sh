for seed in 0 1 2
do
	python prg_discrete.py --seed $seed
	python prg_discrete.py --soft --seed $seed
done