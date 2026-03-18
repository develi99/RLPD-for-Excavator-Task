import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.hidden_dims = (512, 512)

    # Latent-Dimensions
    config.latent_dim_pixels = 128
    config.latent_dim_state = 32

    # for 100x100 -> 7x7 Stein (4x4) ca.0.25 pixel for the stone at outputmap
    # config.cnn_features = (16, 32, 64, 128, 32)
    # config.cnn_filters = (3, 3, 3, 3, 1)
    # config.cnn_strides = (2, 2, 2, 2, 1)

    # 7x7x128
    # config.cnn_features = (16, 32, 64, 128)
    # config.cnn_filters = (3, 3, 3, 3)
    # config.cnn_strides = (2, 2, 2, 2)


    # for 100x100 -> 13x13 Stein (4x4) ca.0.5 pixel for the stone at outputmap
    config.cnn_features = (16, 32, 64, 32)# (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 1)
    config.cnn_strides = (2, 2, 2, 1)

    # with less feature
    # config.cnn_features = (16, 32, 64, 8)# (32, 64, 128, 256)
    # config.cnn_filters = (3, 3, 3, 1)
    # config.cnn_strides = (2, 2, 2, 1)

    # this would have one pixel for input 100x100 and only 10 K Features output, which is feasible (25x25x16=10K)
    # config.cnn_features = (16, 32, 16)
    # config.cnn_filters = (3, 3, 1) # efficient 1x1 filter! Decrease amount of features for efficiency
    # config.cnn_strides = (2, 2, 1)
    # config.cnn_padding = "SAME"

    # Encoder-Typ
    config.encoder = "d4pg"

    return config
