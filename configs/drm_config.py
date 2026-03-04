import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # =========================
    # Agent
    # =========================
    # config._target_ = "agents.drm.DrMAgent"
    config.model_cls = "DrMAgent"
    config.obs_shape = None
    config.action_shape = None
    config.device = "cuda"
    config.lr = 1e-4
    config.feature_dim = 50

    # Critic / Value
    config.critic_target_tau = 0.01
    config.expectile = 0.9

    # Dormant Ratio Mechanism (DrM)
    config.dormant_threshold = 0.025
    config.target_dormant_ratio = 0.2
    config.dormant_temp = 10
    config.lambda_temp = 50
    config.dormant_perturb_interval = 100000
    config.min_perturb_factor = 0.2
    config.max_perturb_factor = None
    config.perturb_rate = 2

    # Exploration
    config.num_expl_steps = 2000
    config.stddev_type = "awake"
    config.stddev_schedule = None
    config.stddev_clip = 0.3

    # Architecture
    config.hidden_dim = 1024
    

    config.target_lambda = 0.6
    config.use_tb = False

    # =========================
    # Task
    # =========================
    # config.task = "manipulator_bring_ball"
    # config.frame_stack = 3
    # config.action_repeat = 2
    # config.discount = 0.99

    # =========================
    # Training
    # =========================
    # config.num_seed_frames = 4000
    # config.update_every_steps = 2
    # config.batch_size = 256
    # config.nstep = 3

    # =========================
    # Evaluation
    # =========================
    # config.eval_every_frames = 10000
    # config.num_eval_episodes = 10

    # =========================
    # Replay Buffer
    # =========================
    # config.replay_buffer_size = 1_000_000
    # config.replay_buffer_num_workers = 4

    # =========================
    # Logging / Misc
    # =========================
    # config.use_tb = True
    # config.use_wandb = True
    # config.save_snapshot = False
    # config.save_video = True
    # config.save_train_video = False
    # config.seed = 121
    # config.experiment = "exp"

    return config