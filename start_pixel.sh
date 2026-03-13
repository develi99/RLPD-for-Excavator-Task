# For training with small network (25x25x16) Feature Vektor
# XLA_PYTHON_CLIENT_PREALLOCATE=false uv run train_finetuning_pixels_excuvator.py --memory_efficient_replay_buffer True --utd_ratio=2 --start_training 5000 --max_steps 800000 --config.backup_entropy=False --config.num_min_qs=2 --config.num_qs=2 --log_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run1/checkpoints --name Reward3 --agxreward 3 --image_size 100 --seed 0 --num_stack 1

# For training with bigger network (13x13x32) Feature Vektor, coarse 
# XLA_PYTHON_CLIENT_PREALLOCATE=false uv run train_finetuning_pixels_excuvator.py --memory_efficient_replay_buffer True --utd_ratio=2 --start_training 5000 --max_steps 800000 --config.backup_entropy=False --config.num_min_qs=2 --config.num_qs=2 --log_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run1/checkpoints --name Reward3 --agxreward 3 --image_size 100 --seed 0 --num_stack 1

# With less Features (13x13x8)
# XLA_PYTHON_CLIENT_PREALLOCATE=false uv run train_finetuning_pixels_excuvator.py --memory_efficient_replay_buffer True --utd_ratio=2 --start_training 5000 --max_steps 800000 --config.backup_entropy=False --config.num_min_qs=2 --config.num_qs=2 --log_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run2/checkpoints --name Reward3 --agxreward 3 --image_size 100 --seed 0 --num_stack 1
# XLA_PYTHON_CLIENT_PREALLOCATE=false uv run train_finetuning_pixels_excuvator.py --memory_efficient_replay_buffer True --utd_ratio=2 --start_training 5000 --max_steps 800000 --config.backup_entropy=False --config.num_min_qs=2 --config.num_qs=2 --log_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run2/checkpoints --name Reward3 --agxreward 3 --image_size 100 --seed 1 --num_stack 1

# (Many Features 7x7x64) for 100x100 -> 7x7 Stein (4x4) ca.0.25 pixel for the stone at outputmap
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run train_finetuning_pixels_excuvator.py --memory_efficient_replay_buffer True --utd_ratio=2 --start_training 5000 --max_steps 800000 --config.backup_entropy=False --config.num_min_qs=2 --config.num_qs=2 --log_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run3/checkpoints --name Reward3 --agxreward 3 --image_size 100 --seed 1 --num_stack 1


# ResNet 


# DrM
# uv run train_finetuning_pixels_excuvator.py --utd_ratio=1 --start_training 5000 --max_steps 800000 --config=configs/drm_config.py --log_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run2/checkpoints --name Reward3 --agxreward 3 --seed 0


# eval example 
# uv run eval_policy.py --save_dir /home/elias/Masterstudium/DeepLearningRobotics/rlpd/logs/pixel/run1/checkpoints/checkpoints --step 150000 --jax --episodes 25 --reward 3 --action-repeat 2 --pixel --image-size 64 --num-stack 3
