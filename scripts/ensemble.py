# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import DataLoader
import pickle
import torch
import os
from model_run import get_data_validation, get_final_data
from utils_study_res import predict, get_model, get_data
from paths import MODEL_PATH

def get_models(runs_id: list):
    """
    Función que dada una lista de runs ids nos de una lista de con los 
    parámetros y una lista de los modelos correspondientes.
    
    Parameters
    ----------
    runs_id: una lista con las ids de las run de los modelos que queremos cargar

    Return
    ----------
    models_list: Lista con los modelos
    params_list: Lista con los parametros de los modelos
    """
    params_list = []
    models_list = []
    
    for run_id in runs_id:
        params, model = get_model(run_id)
        params_list.append(params)
        models_list.append(model)
    return models_list, params_list

def get_data_list(params_list: list):
    """
    Función que dado una lista con los parametros de los distintos modelos
    obtiene los dataloader de train y val para los correctos para cada modelo
    """
    train_loader_list = []
    val_loader_list = []
    train_all_loader_list = []
    test_loader_list = []
    for params in params_list:
        train_loader, val_loader, train_all_loader, test_loader = get_data(params)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
        train_all_loader_list.append(train_all_loader)
        test_loader_list.append(test_loader)
    output = {
        "train": train_loader_list,
        "val": val_loader_list,
        "all_train": train_all_loader_list,
        "test": test_loader_list
    }
    return output

def models_evaluation(loaders: dict, models: list):
    """
    Función que dadlo una lista con modelos y los loader de train y validación
    re
    """
    train_probs_list = []
    test_probs_list = []
    train_labels_list = []
    test_labels_list = []
    # zip con las listas de los dataloader y modeles
    zipped_data_loader = zip(loaders["all_train"], loaders["test"], models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hacemos la predicción para cada modelo con sus correspondietentes imágenes
    # notar que ahora el conjunto de validación es train y el de test es el 
    # conjunto de test
    for train_loader, test_loader, model in zipped_data_loader:
        train_probs, train_labels, _ = predict(model, train_loader)
        test_probs, test_labels, _ = predict(model, test_loader)
        # añadimos a listas sobre las que iteraremos las probalidades y 
        # etiquetas
        train_probs_list.append(train_probs)
        test_probs_list.append(test_probs)
        train_labels_list.append(train_labels)
        test_labels_list.append(test_labels)
    return train_probs_list, test_probs_list, train_labels_list, test_labels_list

def metalearner(train_probs, test_probs, train_labels, test_labels, model: str,
                seed=123):
    """
    Función donde lo que hacemos es entrenar el metalearner
    """
    n_learners = len(train_probs)
    n_instances_tr = train_probs[0].shape[0]
    n_instances_tst = test_probs[0].shape[0]
    dataset = zip(train_probs, test_probs, train_labels, test_labels)
    X_train = np.zeros([n_instances_tr, n_learners])
    X_test = np.zeros([n_instances_tst, n_learners])
    i = 0
    for train_prob, test_prob, train_label, test_label in dataset:
        X_train[:, i] = train_prob[:, 0]
        y_train = train_label
        X_test[:, i] = test_prob[:, 0]
        y_test = test_label
        i += 1

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    if model == "RandomForest":
        # definimos el modelo con búsqueda de hiperparámetros y con 
        # validación cruzada
        param_grid = {
            "n_estimators": [1, 10],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 1, 2, 5, 10, 100]
        }
        # la definimos aquí para tener reproducibilidad
        clf = RandomForestClassifier()
        cv_grid_search(X_train, y_train, X_test, y_test, kf, param_grid, clf, model)
    elif model == "lr":
        clf = LogisticRegression()
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 10]
        }
        cv_grid_search(X_train, y_train, X_test, y_test, kf, param_grid, clf, model)
    else:
        raise Exception("model not defined")

def cv_grid_search(X_train, y_train, X_test, y_test, kf, param_grid, clf, model):
    """
    Función que lo que hace es hacer el grid search sobre el grid dado y el 
    clasificador dado. También se tienen que dar las particiones de 
    de validación cruzada así como los conjuntos de entrenamiento y test. 
    Una vez obtenidos los mejors parametros se entrna un único clf que 
    se logea en optuna
    """
    gs = GridSearchCV(
            clf,
            param_grid=param_grid,
            scoring="accuracy",
            cv=kf,
            n_jobs=-1,
            verbose=1
        )
    gs_results = gs.fit(X_train, y_train)
    best_params = gs_results.best_params_
    print("Accuracy grid search:", gs_results.best_score_)
    acc_test = gs.score(X_test, y_test)
    mlflow.log_param("param_gris", gs.param_grid)
    print("Accuracy test:", acc_test)
        # Entrenamos el mejor modelo
    print("Entrenado el mejor modelo")
    # with mlflow.start_run(nested=True):
    if model == "lr":
        clf = LogisticRegression(**best_params)
    elif model == "RandomForest":
        clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    mlflow.log_metric("acc_train", acc_train) 
    mlflow.log_metric("acc_test", acc_test)
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(clf, "model")

def ensemble_pipeline(runs_id: list, experiment_name: str, model="RandomForest"):
    mlflow.set_tracking_uri(MODEL_PATH)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model_list, params_list = get_models(runs_id)
        loaders = get_data_list(params_list)
        train_probs, test_probs, train_labels, test_labels = models_evaluation(
            loaders, model_list
        )
        metalearner(
            train_probs, test_probs, train_labels, test_labels, model=model
        )
        log_ids(runs_id)

def log_ids(runs_id):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mlflow.log_param(f"_id", timestamp)
    runs_id_dict = {}
    for index, run_id in enumerate(runs_id):
        runs_id_dict[f"_run_id_{index}"] = run_id
    mlflow.log_params(runs_id_dict)
    with open("temp" + "/runs_id", "wb") as file:
            pickle.dump(runs_id, file)
    mlflow.log_artifact("temp" + "/runs_id")
# %%
if __name__ == "__main__":
    exp_name = "Ensemble proyecciones"
    runs_id = ["fc01bea6beed4ea3a207355aab74b5c5", "1797915b1f7947eba1ea80dac992b42b"]
    ensemble_pipeline(runs_id, exp_name, "lr")

# %%
