# for seed in 1 2
for seed in 0 1 2
do
	# python prg_discrete.py --rs 100 --seed 0 --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu --seed $seed --target_action
	# python prg_discrete.py --rs 100 --seed 0 --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu --seed $seed
	# python masac_discrete.py --rs 100 --seed 0 --num_agent 2 --epoch 300 --exp_name 2pHard --port 9394 --gpu --seed $seed --online_action
	# python masac_discrete.py --rs 100 --seed 0 --num_agent 2 --epoch 300 --exp_name 2pHard --port 9394 --gpu --seed $seed
	# python masac_discrete.py --rs 100 --num_agent 3 --epoch 300 --seed $seed --exp_name 3pHard --port 9395 --gpu
	# python masac_discrete.py --rs 100 --num_agent 3 --epoch 300 --seed $seed --exp_name 3pHard --port 9396 --gpu --online_action
	python prg_discrete.py --rs 100 --seed 0 --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu --seed $seed --online_action --use_gumbel
done