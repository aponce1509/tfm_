import torch
import torchvision.transforms as T

# Model
model_params = {
    "model_name": "convnet",
    "input_shape": (62, 128),
    "conv_filters": (16, 24),
    'bn': False,
    "conv_sizes": (3, 3),
    "conv_strides": (1, 1),
    "pool_sizes": (2, 2),
    "pool_strides": (2, 2),
    "conv_no_linear_fun": "relu",
    "clf_n_layers": (3),
    "clf_neurons": (32, 8, 2),
    'n_channels': 3,
    "clf_no_linear_fun": "relu",
    "dropout": 0.0,
    "optim_name": "adam",
    "optim_lr": 0.001,
    "scheduler_name": "steplr",
    "scheduler_step_size": 3,
    "scheduler_gamma": 0.7,
    "criterion_name": "nllloss",
    'avg_layer': "identity",
    'avg_layer_size': None,  # (14, 30)
    "batch_size": 48,
    "n_epochs": 10,
}
params = {
    "seeds": (19, ),
    "seed_": 3,
    # parámetros de datos
    "cube_shape_x": 1000,
    "win_shape": (62, 62, 128),
    "projection": "color",
    "projection_pool": "max",
    "cube_pool": "mean",
    "log_trans": False,
    "transform_des": "Normalización",
    # "transform": None,
    "transform": T.Compose([
        T.Lambda(lambda x: x.type(torch.float)),
        T.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    "is_fast": True,
    # parámetros de entrenamiento de la red
    "is_def": False,
}

grid = {"grid": {
    "optimizer_grid": {
        "name": "optim_name",
        "choices": ["adam", "adadelta"]
    },
    "lr_grid": {
        "name": "lr",
        "low": 1e-4,
        "high": 1e-1
    }
}
}

params = {**params, **model_params, **grid}