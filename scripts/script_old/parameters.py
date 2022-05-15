import torchvision.transforms as transforms

# Variables globales
IMG_PATH = "D:\\data/"

# Model
model_params = {
    "model_name": "convnet",
    "input_shape": (62, 128),
    "n_filters": (16, 24),
    "pool_size": 2,
    "kernel_size": 3,
    "dropout": 0.2,
    "optim_name": "adam",
    "optim_lr": 0.001,
    "scheduler_name": "steplr",
    "scheduler_step_size": 1,
    "scheduler_gamma": 0.7,
    "criterion_name": "crossentropyloss"
}
params_test = {
    "seed_": 123,
    # parámetros de datos
    "cube_shape_x": 1000,
    "win_shape": (62, 62, 128),
    "projection": "y",
    "projection_pool": "max",
    "cube_pool": "mean",
    "transform": None,
    "is_fast": True,
    # parámetros de entrenamiento de la red
    "batch_size": 32,
    "n_epochs": 20,
    "is_def": False,
    # parámetros de optuna ?
    "n_trials": 5
}

params_test = {**params_test, **model_params}