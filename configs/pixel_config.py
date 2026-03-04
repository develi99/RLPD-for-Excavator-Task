import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.hidden_dims = (256, 256)

    # Latent-Dimensionen
    config.latent_dim_pixels = 64
    config.latent_dim_state = 16

    # CNN für Pixel-Eingaben 128x128x9 (3 gestapelte Bilder)
    config.cnn_features = (16, 32, 64, 16)# (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 1)
    config.cnn_strides = (2, 2, 2, 1)
    config.cnn_padding = "SAME"

    # Encoder-Typ
    config.encoder = "d4pg"

    return config
