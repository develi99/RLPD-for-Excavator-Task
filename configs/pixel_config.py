import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.hidden_dims = (512, 512)

    # Latent-Dimensions
    config.latent_dim_pixels = 64
    config.latent_dim_state = 16

    # for 100x100 -> 13x13 Stein (4x4) ca.0.5 pixel for the stone at outputmap
    config.cnn_features = (16, 32, 64, 32)# (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 1)
    config.cnn_strides = (2, 2, 2, 1)

    # this would have one pixel for input 100x100 and only 10 K Features output, which is feasible (25x25x15=10K)
    # config.cnn_features = (16, 32, 16)
    # config.cnn_filters = (3, 3, 1) # efficient 1x1 filter! Decrease amount of features for efficiency
    # config.cnn_strides = (2, 2, 1)
    # config.cnn_padding = "SAME"

    # for 256x256 -> 16x16 Stein (5x5) less than 1 pixel for the stone at outputmap
    # try same as above for fine grained feature map
    # config.cnn_features = (16, 32, 64, 128, 32)# (32, 64, 128, 256)
    # config.cnn_filters = (3, 3, 3, 3, 1)
    # config.cnn_strides = (2, 2, 2, 2, 1)
    # config.cnn_padding = "SAME"

    # Encoder-Typ
    config.encoder = "d4pg"

    return config
