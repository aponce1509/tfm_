from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import pickle
import torch.nn.functional as F
import os
import pandas as pd
from paths import MODEL_PATH, TORCH_PATH, cuda_device
from ensemble import get_models, get_data_list
from utils_study_res import predict, get_model, metrics_calculation, efficiency_purity_energy
import matplotlib.pyplot as plt
import seaborn as sns


def models_evaluation_ensemble(loaders: dict, models: list):
    """
    Función que dadlo una lista con modelos y los loader de train y validación
    re
    """
    train_probs_list = []
    test_probs_list = []
    train_labels_list = []
    test_labels_list = []
    train_en_list = []
    test_en_list = []
    # zip con las listas de los dataloader y modeles
    zipped_data_loader = zip(loaders["all_train"], loaders["test"], models)
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    # hacemos la predicción para cada modelo con sus correspondietentes imágenes
    # notar que ahora el conjunto de validación es train y el de test es el 
    # conjunto de test
    for train_loader, test_loader, model in zipped_data_loader:
        train_probs, train_labels, train_en = predict(model, train_loader)
        test_probs, test_labels, test_en = predict(model, test_loader)
        # añadimos a listas sobre las que iteraremos las probalidades y 
        # etiquetas
        train_probs_list.append(train_probs)
        test_probs_list.append(test_probs)
        train_labels_list.append(train_labels)
        test_labels_list.append(test_labels)
        train_en_list.append(train_en)
        test_en_list.append(test_en)
    return train_probs_list, test_probs_list, train_labels_list, test_labels_list, train_en_list, test_en_list

def get_data_ensemble(runs_id, transforms=None):
    model_list, params_list = get_models(runs_id)
    loaders = get_data_list(params_list, transforms)
    train_probs, test_probs, train_labels, test_labels, train_ens, test_ens = models_evaluation_ensemble(
        loaders, model_list
    )
    n_learners = len(train_probs)
    n_instances_tr = train_probs[0].shape[0]
    n_instances_tst = test_probs[0].shape[0]
    dataset = zip(train_probs, test_probs, train_labels, test_labels, train_ens, test_ens)
    X_train = np.zeros([n_instances_tr, n_learners])
    X_test = np.zeros([n_instances_tst, n_learners])
    print(train_probs[0].shape)
    i = 0
    for train_prob, test_prob, train_label, test_label, en_tr, en_tst in dataset:
        X_train[:, i] = train_prob[:, 0]
        # print(X_train.shape)
        y_train = train_label
        X_test[:, i] = test_prob[:, 0]
        y_test = test_label
        energy_train = en_tr
        energy_test = en_tst
        i += 1
    return X_train, X_test, y_train, y_test, energy_train, energy_test

def predict_ensemble(model, X):
    probs = model.predict_proba(X)
    return probs

def get_metrics_ensemble(save, model, X, y_true, energy, case_of_study, n_bins):
    probs = predict_ensemble(model, X)
    _ = metrics_calculation(probs, y_true)
    _ = efficiency_purity_energy(probs, y_true, energy, case_of_study, n_bins,
                                 save)

def metric_study_ensemble(run_id: str, case_of_study, transforms=None,
                          save: bool=False, n_bins=50):
    runs_id, model = get_model(run_id, is_enesemble=True)
    X_train, X_test, y_train, y_test, energy_train, energy_test = get_data_ensemble(runs_id, transforms)    
    if case_of_study == "train":
        get_metrics_ensemble(save, model, X_train, y_train, energy_train,
                             case_of_study, n_bins)
    elif case_of_study == "test":
        get_metrics_ensemble(save, model, X_test, y_test, energy_test,
                             case_of_study, n_bins)
    else:
        raise Exception("Not valid case_of_study")

if __name__ == "__main__":
    run_id = "6f992dbc4d894711ac9841e7b9d3ed2a"
    study = "train"
    metric_study_ensemble(run_id, study, False, 50)