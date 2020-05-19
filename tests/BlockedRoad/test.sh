for seed in 0
do
	# python masac_discrete.py --rs 100 --seed $seed --num_agent 3 --epoch 300 --exp_name 3pHard --port 9394 --gpu
	# python prg_discrete.py --rs 100 --double_q --seed $seed --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu
	# python prg_discrete.py --rs 100 --double_q --seed $seed --num_agent 3 --epoch 300 --exp_name 3pHard --port 9394 --gpu
	# python maddpg_gumbel.py --double_q --seed $seed --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu
	python maddpg_gumbel.py --double_q --online_action --seed $seed --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu
	# python maddpg_gumbel.py --double_q --seed $seed --num_agent 3 --epoch 300 --exp_name 3pHard --port 9394 --gpu
	# python prg_gumbel.py --double_q --seed $seed --num_agent 2 --epoch 300 --exp_name 2pHard --port 9394 --gpu
	# python prg_gumbel.py --double_q --entropy --rs 10.0 --seed $seed --num_agent 2 --epoch 300 --exp_name 2pHard --port 9393 --gpu
done