from ml_collections.config_dict import config_dict
from configs import drq_config
import ml_collections

# conv1: (2, 112, 112, 64)
# block1_0: (2, 56, 56, 64)
# block1_1: (2, 56, 56, 64)
# block2_0: (2, 28, 28, 128)
# block2_1: (2, 28, 28, 128)
# block3_0: (2, 14, 14, 256)
# block3_1: (2, 14, 14, 256)
# block4_0: (2, 7, 7, 512)
# block4_1: (2, 7, 7, 512)

def get_config():
    config = ml_collections.ConfigDict()

    config.hidden_dims = (512, 512)

    config.model_cls = "FeatureDrQLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.discount = 0.99

    config.num_qs = 2

    config.tau = 0.005
    config.init_temperature = 0.1
    config.backup_entropy = True
    config.target_entropy = config_dict.placeholder(float)


    config.num_qs = 10
    config.num_min_qs = 1

    config.critic_layer_norm = True
    config.backup_entropy = False

    return config
