from random import choices
import torchvision.transforms as transforms

# Model
model_params = {
    "model_name": "convnet_big",
    "input_shape": (128, 224),
    "n_filters": (16, 32),
    "pool_size": 2,
    "kernel_size": (5, 5),
    "kernel_stride": (1, 1),
    "dropout": 0.2,
    "optim_name": "adam",
    "optim_lr": 0.001,
    "scheduler_name": "steplr",
    "scheduler_step_size": 1,
    "scheduler_gamma": 0.7,
    "criterion_name": "crossentropyloss"
}
params = {
    "seed_": 1,
    # parámetros de datos
    "cube_shape_x": 1000,
    "win_shape": (128, 128, 224),
    "projection": "y",
    "projection_pool": "max",
    "cube_pool": "mean",
    "transform": None,
    "is_fast": True,
    # parámetros de entrenamiento de la red
    "batch_size": 32,
    "n_epochs": 50,
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