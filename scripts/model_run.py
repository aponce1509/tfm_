# %%
# functions for model run
import mlflow
from datetime import datetime
import optuna
import pickle
import random
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torch import optim
from paths import MODEL_PATH, TORCH_PATH, cuda_device
import torch.nn as nn
from torch.utils.data import DataLoader

from utils_win_cube_copy import Cascadas, CascadasFast, CascadasMulti, CascadasMultiEff
from _models import get_gradient_elements, get_criterion
# from parameters import params_test

def fix_seed(params: dict, pytorch_alg=True):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(pytorch_alg)
    random.seed(params["seed_"])
    torch.manual_seed(params["seed_"])
    np.random.seed(params["seed_"])

# Pipelines

def objective(trial: optuna.Trial, params: dict, opt_id=None,
              experiment_name="Defalut"):
    hyperparameter_suggest(trial, params)
    # fix_seed(params)
    error = basic_pipeline(params, is_optuna=True, optuna_step=trial.number,
                           verbose=False, trial_params=trial.params,
                           opt_id=opt_id, experiment_name=experiment_name,
                           trial=trial)
    return error

def optuna_pipeline(params: dict, experiment_name: str, n_trials: int=5):
    """
    Pipeline de la optimización de hiperparámetros usando optuna.
    """
    # nos metemos en el experimento y optenemos un id para la run de optuna
    mlflow.set_tracking_uri(MODEL_PATH)
    mlflow.set_experiment(experiment_name)
    optuna_name = get_optuna_name()
    with mlflow.start_run(run_name=optuna_name):
        sampler = optuna.samplers.TPESampler(seed=params["seed_"])
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5,
                                             n_warmup_steps=5)
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    pruner=pruner)
        study.optimize(
            lambda trial: objective(trial, params, optuna_name, experiment_name),
            n_trials=n_trials
            )
        print_study_stats(study)

def basic_pipeline(params, is_optuna=False, optuna_step=None,
                   verbose=True, trial_params=None, experiment_name="Default",
                   opt_id=None, trial=None):
    fix_seed(params)
    mlflow.set_tracking_uri(MODEL_PATH)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(nested=True):
        model, device, optimizer, scheduler, criterion = get_gradient_elements(
            params
        )
        train_data, val_data = get_data_validation(params)

        val_loss, metrics, _id = train_loop(
            train_data=train_data,
            val_data=val_data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            params=params,
            verbose=verbose,
            is_optuna=is_optuna,
            optuna_id=opt_id,
            trial=trial
        )
        mlflow_log(params, is_optuna, metrics, _id)

    mlflow_log_optuna(is_optuna, trial_params, val_loss, optuna_step)

    return val_loss

def final_test_pipeline(params: dict):
    mlflow.set_experiment("final_tests")
    with mlflow.start_run(nested=True):
        train_data, test_data = get_final_data(params)
        
        model, device, optimizer, scheduler, criterion = get_gradient_elements(
            params
        )

        val_loss, metrics, _id = train_loop(
            train_data=train_data,
            val_data=test_data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            params=params
        )

        mlflow_log(params, False, metrics, _id)

# Functions

def test_model(run_id, model_log):
    """
    Función que dados los parámetros y el log del modelo de mlflow, obtiene el
    accuracy en el conjunto de prueba.
    """
    # cargamos el artifact con los parametros
    client = mlflow.tracking.MlflowClient()
    if not os.path.exists('temp'):
        os.mkdir('temp')
    local_dir = "temp/downloads"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    local_path = client.download_artifacts(run_id, "params", local_dir)
    with open(local_path, "rb") as file:
        params = pickle.load(file)

    mlflow.set_experiment("Test validation simple")
    # empezamos la run
    with mlflow.start_run():
        # subimos lo parametros a mlflow
        mlflow.log_params(params)
        mlflow.set_tag("mlflow.runName", model_log)
        # cargamos los datos de test
        print(f"Testing model: {model_log}")
        test_data = get_data_test(params)
        test_loader = DataLoader(test_data, params["batch_size"], shuffle=True)
        print(f'Train set has {len(test_data)} instances')
        # cargamos el modelo preentrenado
        loaded_model = mlflow.pytorch.load_model(model_log)
        device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        loaded_model.to(device)
        # hacemos la validación con los datos de test
        test_metrics = validation(
            model=loaded_model,
            device=device,
            validation_loader=test_loader,
            criterion=get_criterion(params)
        )
        test_metrics = {
            "test_loss": test_metrics[0],
            "test_acc": test_metrics[1]
        }
        mlflow.log_metrics(test_metrics)
    return test_metrics

def get_data_validation(params: dict):
    if not 'params_multi' in params:
        if params['model_name'][0:6] == 'concat':
            train_dataset = CascadasMulti(
                train=True,
                validation=True,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            val_dataset = CascadasMulti(
                train=False,
                validation=True,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            return train_dataset, val_dataset
        if not params["is_fast"]: 
            train_dataset = Cascadas(
                train=True,
                validation=True,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            val_dataset = Cascadas(
                train=False,
                validation=True,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
        else:
            train_dataset = CascadasFast(
                train=True,
                validation=True,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            val_dataset = CascadasFast(
                train=False,
                validation=True,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
    else:
        seed = params['seed_']
        transform = params['transform_multi']
        params_multi = params['params_multi']
        train_dataset = CascadasMultiEff(params_multi, seed, True, False,
                                         transform)
        val_dataset = CascadasMultiEff(params_multi, seed, True, True,
                                       transform)

    return train_dataset, val_dataset

def get_data_test(params: dict):
    if params['model_name'][0:6] == 'concat':
        test_data = CascadasMulti(
            train=False,
            validation=False,
            seed_=params["seed_"],
            cube_shape_x=params["cube_shape_x"],
            win_shape=params["win_shape"],
            cube_pool=params["cube_pool"],
            projection_pool=params["projection_pool"],
            transform=params["transform"],
            log_trans=params['log_trans']
        )
        return test_data
    if not params["is_fast"]:
        test_data = Cascadas(
            train=False,
            validation=False,
            seed_=params["seed_"],
            cube_shape_x=params["cube_shape_x"],
            win_shape=params["win_shape"],
            projection=params["projection"],
            cube_pool=params["cube_pool"],
            projection_pool=params["projection_pool"],
            transform=params["transform"],
            log_trans=params['log_trans']
        )
    else:
        test_data = CascadasFast(
            train=False,
            validation=False,
            seed_=params["seed_"],
            cube_shape_x=params["cube_shape_x"],
            win_shape=params["win_shape"],
            projection=params["projection"],
            cube_pool=params["cube_pool"],
            projection_pool=params["projection_pool"],
            transform=params["transform"],
            log_trans=params['log_trans']
        )
    return test_data

def get_final_data(params: dict):
    if not 'params_multi' in params:
        if params['model_name'][0:6] == 'concat':
            train_dataset = CascadasMulti(
                train=True,
                validation=False,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            test_dataset = CascadasMulti(
                train=False,
                validation=False,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            return train_dataset, test_dataset
        if not params["is_fast"]:
            train_dataset = Cascadas(
                train=True,
                validation=False,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            test_dataset = Cascadas(
                train=False,
                validation=False,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
        else:
            train_dataset = CascadasFast(
                train=True,
                validation=False,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
            test_dataset = CascadasFast(
                train=False,
                validation=False,
                seed_=params["seed_"],
                cube_shape_x=params["cube_shape_x"],
                win_shape=params["win_shape"],
                projection=params["projection"],
                cube_pool=params["cube_pool"],
                projection_pool=params["projection_pool"],
                transform=params["transform"],
                log_trans=params['log_trans']
            )
    else:
        seed = params['seed_']
        transform = params['transform_multi']
        params_multi = params['params_multi']
        train_dataset = CascadasMultiEff(params_multi, seed, True, False,
                                         transform)
        test_dataset = CascadasMultiEff(params_multi, seed, False, False,
                                       transform)
    return train_dataset, test_dataset

def train(model: nn.Module, optimizer: optim.Optimizer, device: str, 
          train_loader: DataLoader, writer: SummaryWriter, criterion,
          epoch: int, verbose=True):
    model.train()
    running_loss = 0.
    total_loss = 0.
    total_correct = 0
    n_batches = len(train_loader)
    # gradiente descendiente por bach
    for batch_idx, (inputs, labels, _) in enumerate(train_loader):
        # paso del gradiente
        optimizer.zero_grad()
        if type(inputs) == tuple or type(inputs) == list:
            inputs = [input_.to(device, dtype=torch.float) for input_ in inputs]
            labels = labels.to(device)
            outputs = model(*inputs)
        else:
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # modificamos los errores uno para la epoca completa y otro que guarda
        # solo una parte final
        running_loss += loss.item()
        total_loss += loss.item()
        # Hacemos que se muestren 10 veces en pantalla por época
        batchs_per_print = int(n_batches / 10)
        # obtenemos acc
        predicted = torch.max(outputs.data, 1)[1]
        total_correct += sum(predicted == labels).item()

        if (batch_idx % batchs_per_print == batchs_per_print - 1):
            # mean loss in the curren group of batchs
            last_loss = running_loss / batchs_per_print
            if verbose:
                print(f'  batch {batch_idx + 1} loss: {last_loss:.3g}') 
            # tensorboard
            tb_x = epoch * len(train_loader) + batch_idx + 1  # eje x
            writer.add_scalar('Loss/train', last_loss, tb_x)
            # rest the running_loss
            running_loss = 0.
    
    train_size = len(train_loader.dataset)
    # obtenemos métricas
    train_acc = total_correct / train_size
    # es el error medio de todo el conjunto de datos
    mean_loss_train = total_loss / (batch_idx + 1)
    return mean_loss_train, train_acc

def validation(model: nn.Module, device: str, 
               validation_loader: DataLoader, criterion):
    """
    Función que dado un modelo y un conjunto de datos (de la clase Dataloader)
    nos devuelve el valor medio de los valores de la función loss y el porcentaje
    de acierto.
    Parameter
    ---------
    model: Modelo a validar
    device: Device donde esta cargado el modelo
    validation_loader: loader de conjunto que se quier validar
    criterion: Criterio que se usa para obtener la función pérdida (solo esta
    probada con la crossentropy)
    Return
    ----------
    mean_val_loss: valor medio de los valores de perdida
    val_acc: porcentaje de acierto del conjunto de validación
    """
    model.eval()
    val_size = len(validation_loader.dataset)
    val_loss = 0.
    total_correct_val = 0
    # entreamos en el modelo sin modificar sus valores
    with torch.no_grad():
        # iteramos por batchs
        for batch_idx, (inputs, labels, _) in enumerate(validation_loader):
            labels = labels.to(device)
            if type(inputs) == tuple or type(inputs) == list:
                inputs = [input_.to(device, dtype=torch.float) for input_ in inputs]
                outputs = model(*inputs)
            else:
                inputs = inputs.to(device, dtype=torch.float)
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.max(outputs.data, 1)[1]
            total_correct_val += sum((predicted == labels)).item()

    mean_val_loss = val_loss / (batch_idx + 1)
    val_acc = total_correct_val / val_size
    return mean_val_loss, val_acc

def train_loop(train_data, val_data, model, optimizer, criterion,
                scheduler, device, params: dict,
               verbose=True, is_optuna=False, optuna_id=None, trial=None):
    """
    Función donde se realiza el entrenamiento de la red por las distintas épocas
    """
    print("Cargando datos:")

    # load data and init the train loop
    train_loader = DataLoader(train_data, params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, params["batch_size"], shuffle=True)
    print(f'Training set has {len(train_data)} instances')
    print(f'Validation set has {len(val_data)} instances')
    # Definición elementos de la red neuronal
    # gpu
    print("Device:", device)
    # configuramos tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if params["is_def"]:
        writer = SummaryWriter(f'runs/def_{params["model_name"]}_{timestamp}')
    else:
        writer = SummaryWriter(f'runs/temp_{params["model_name"]}_{timestamp}')
    # si estamos en un modelo de optuna que se guarde en otra carpeta
    if is_optuna:
        params["_id_optuna"] = optuna_id
        writer = SummaryWriter(
            f'runs/opt_{optuna_id}/{params["model_name"]}_{timestamp}'
        )    
    # Empezamos con el entrenamiento
    print("training:")
    print(f"timestamp: {timestamp}")
    best_val_loss = 1_000_000
    best_val_acc = 0
    early_stop = EarlyStop()
    
    # epochs loop
    for epoch in range(params["n_epochs"]):
        print(f"EPOCH {epoch + 1}:")
        train_metrics = train(
            model=model,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            writer=writer,
            epoch=epoch,
            verbose=verbose            
        )
        val_metrics = validation(
            model=model,
            device=device,
            validation_loader=val_loader,
            criterion=criterion
        )
        # TODO if params["is_def"] entonces el earlystop se tiene que hacer 
        # solo mirando train ES DELICADO YA QUE NO PUEDES VER SI HAY OVERFITING
        is_stop = early_stop.step(
            best_val_acc, val_metrics[1], train_metrics[1]
        )
        print(f'Loss train: {train_metrics[0]:.3g}' +
              f' validation {val_metrics[0]:.3g}')
        print(f'Acc train {train_metrics[1]:.3g}' + 
              f' validation: {val_metrics[1]:.3g}')
        # añadimos a tensorboard las métricas
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training' : train_metrics[0], 'Validation' : val_metrics[0]},
            epoch + 1
        )
        writer.add_scalars(
            'Training vs. Validation Accuracy',
            {'Training' : train_metrics[1], 'Validation' : val_metrics[1]},
            epoch + 1
        )
        writer.flush()

        if val_metrics[1] > best_val_acc:
            best_val_acc = val_metrics[1]
            # CAMBIAR FORMATO
            model_path = f'{TORCH_PATH}/models/{params["model_name"]}_{timestamp}'
            torch.save(model.state_dict(), model_path)
            mlflow_model_name = f"{params['model_name']}_{timestamp}"
            mlflow.pytorch.log_model(model, mlflow_model_name)
        if val_metrics[0] < best_val_loss:
            best_val_loss = val_metrics[0]
        if scheduler:
            scheduler.step()
        metrics = {
            "loss_train": train_metrics[0],
            "loss_val": val_metrics[0],
            "acc_train": train_metrics[1],
            "acc_val": val_metrics[1],
            "_best_val_acc": best_val_acc
        }
        mlflow.log_metrics(metrics, epoch)
        # Si estamos en optuna usamos la opción de prune trials
        if is_optuna:
            trial.report(best_val_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        # Paramos el entrenamiento si estamos estancados
        if is_stop:
            print("Early stoping")
            error = best_val_loss
            return error, metrics, timestamp

    # dataiter = iter(train_loader)
    # images, _, _ = dataiter.next()
    # images.to(device)
    # # create grid of images
    # img_grid = torchvision.utils.make_grid(images)
    # # write to tensorboard
    # writer.add_image('one batch', img_grid)
    error = best_val_loss
    return error, metrics, timestamp

def mlflow_log_optuna(is_optuna, trial_params, val_loss, optuna_step):
    """
    Funcíon auxiliar para guardar en mlflow los resultados de optuna
    """
    if is_optuna:
        # mlflow.log_metrics(trial_params, optuna_step)
        mlflow.log_metric("val_loss_optuna", val_loss, optuna_step)

def mlflow_log(params: dict, is_optuna, metrics, _id):
    """
    Función auxiliar para cargar en mlflow los parámetros del modelo 
    """
    # introducimos al dict con los parámetros si estamos en optuna y la
    # _id (el timestamp)
    params["is_optuna"] = is_optuna
    params["_id"] = _id
    # lista con las claves que no queremos registrar
    no_log_keys = ["model", "optimizer", "criterion", "scheduler", "grid", 
                   'transform', 'transform_multi', 'params_multi']
    # los eliminamos de una copia
    params_aux = params.copy()
    # iteramos por las claves y quitamos si está dicha clave
    for key in no_log_keys:
        if key in params_aux:
            del params_aux[key]
    # cargamos en mlflow los parámetros y las métricas
    mlflow.log_params(params_aux)
    mlflow.log_metrics(metrics)
    # creamos un archivo temporal con los parámetros para crear el artifact 
    # con los parámetros
    if not os.path.exists('temp'):
        os.mkdir('temp')
    with open("temp" + "/params", "wb") as file:
            pickle.dump(params_aux, file)
    mlflow.log_artifact("temp" + "/params")

def hyperparameter_suggest(trial: optuna.Trial, params: dict):
    n_conv = len(params['conv_sizes'])
    grid = params["grid"]
    if "optimizer_grid" in grid:
        optim = trial.suggest_categorical(**grid["optimizer_grid"])
        params["optim_name"] = optim
    if "lr_grid" in grid:
        lr = trial.suggest_int(**grid["lr_grid"])
        params["optim_lr"] = 10**(-1*lr)
    if "scheduler_gamma" in grid:
        gamma = trial.suggest_int(**grid["scheduler_gamma"])
        params["scheduler_gamma"] = gamma * 0.1
    if 'kernel_size' in grid:
        k_size = trial.suggest_int(**grid['kernel_size'], step=2)
        params['conv_sizes'] = (k_size, ) * n_conv
    if 'kernel_stride' in grid:
        k_stride = trial.suggest_int(**grid['kernel_stride'], step=1)
        params['conv_strides'] = (k_stride, ) * n_conv
    if 'unfreeze_layers' in grid:
        ul = trial.suggest_categorical(**grid['unfreeze_layers'])
        params['unfreeze_layers'] = ul
    print(trial.params)

def get_optuna_name():
    timestamp_optuna = datetime.now().strftime('%Y%m%d_%H%M%S')
    optuna_name = f"optuna_{timestamp_optuna}"
    return optuna_name

def print_study_stats(study):
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    mlflow.log_params(trial.params)
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

class EarlyStop():
    """
    Clase para que para el entrenamiento de la red si se esta produciendo 
    overfitting.
    Parameters
    ----------
    patience: nº de veces que seguidas antes de que se haga detenga el 
    entrenamiento
    acc_max_diff: diferencia máx entre el acc de train y validación, si 
    esta dif se tiene se para el entrenamiento
    """
    def __init__(self, patience=10, acc_max_diff=0.03) -> None:
        self.patience = patience
        self.acc_max_diff = acc_max_diff
        self.patience_status_acc_diff = 0
        self.patience_status_val = 0
    
    def step(self, best_val_acc, new_val_acc, new_train_acc):
        """
        En cada época se mira si el mejor accuracy de valición es mayor que
        el accuracy obtenido en esa época (hasta la 3 cifra significativa).
        Si ocurre patience veces seguidas devuelve True en caso contrario, 
        False. También se mira si el acc de train es mayor que
        Parameters
        ----------
        best_val_acc: mejor acc de validación
        new_val_acc: Nuevo valor de acc en validación 
        new_train_acc: Nuevo valor de acc en train
        """
        # miramos si el mejor acc de val es menor que el nuevo val acc
        if round(best_val_acc, 3) >= round(new_val_acc, 3):
            self.patience_status_val += 1
        else:
            self.patience_status_val = 0
        print("patience_status_val:", self.patience_status_val)
        acc_diff = new_train_acc - best_val_acc
        if acc_diff > self.acc_max_diff:
            self.patience_status_acc_diff += 1
        else:
            self.patience_status_acc_diff = 0
        if self.patience_status_val == self.patience:
            return True
        elif self.patience_status_acc_diff == self.patience:
            return True
        return False
# %%
if __name__ == "__main__":
    # optuna_pipeline(params_test, experiment_name="Projection_y")
    from exp_p_y.params import params
    params['model_name'] = 'concat'
    params['input_shape'] = ((62, 128), (62, 128), (62, 62))
    params['batch_size'] = 48
    basic_pipeline(params, experiment_name='ConCat')
    pass

# %%
# from utils_win_cube_copy import CascadasFast
# import torch
# cascada = CascadasFast(cube_shape_x=1000, projection="x", 
#                            win_shape=(62, 62, 128))
# cascada.plot_simple(42)
