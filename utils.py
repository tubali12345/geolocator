class Config:
    PATH = 'C:/Users/TuriB/Documents/5.felev/bevadat/geo_project/'


class Parameters:
    num_classes = 50
    num_epochs = 50
    learning_rate = 1e-4
    min_lr = 0
    reduce_lr_factor = 0.2
    reduce_lr_patience = 1
    early_stop_patience = 5
    momentum = 0.9
    verbose = 0
    weight_decay = 0.1
    image_size = 256
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
