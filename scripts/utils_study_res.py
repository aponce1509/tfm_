# %%
# %reload_ext autoreload
# %autoreload 2
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import pickle
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as T
import os
import pandas as pd
from paths import MODEL_PATH, TORCH_PATH, cuda_device
from model_run import get_final_data, fix_seed, get_data_validation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from grad_cam import get_cam

def get_model(run_id, is_enesemble=False):
    """
    Función que dado el id de la run a estudiar nos devuelve el modelo y los
    parámetros de dicho modelo
    Parameters
    ----------
    Return
    ----------
    params: dicicionario con los parámetros del modelo
    model: modelo listo para entrenar en nuevos datos
    """
    client = mlflow.tracking.MlflowClient(MODEL_PATH)
    # cargamos los parámetros
    local_dir = "temp/downloads"  # carpeta donde dejar el artifact
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    mlflow.set_tracking_uri(MODEL_PATH)
    # obtenemos el modelo
    if is_enesemble:
        # cargamos el artifact
        local_path = client.download_artifacts(run_id, "runs_id", local_dir)
        with open(local_path, "rb") as file:
            runs_id = pickle.load(file)
        model_log = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_log)
        return runs_id, model
    else:
        # cargamos el artifact
        local_path = client.download_artifacts(run_id, "params", local_dir)
        with open(local_path, "rb") as file:
            params = pickle.load(file)
        model_log = f"runs:/{run_id}/{params['model_name']}_{params['_id']}"
        model = mlflow.pytorch.load_model(model_log, map_location=cuda_device)
        return params, model

def get_data(params: dict, transform=None):
    """
    Función para obtener los data loader de train validación y de test dado 
    los parámetros del modelo que estamos estudiando
    """
    params_aux = params.copy()
    params_aux['transform'] = transform
    train_dataset, val_dataset = get_data_validation(params_aux)
    train_all_dataset, test_dataset = get_final_data(params_aux)
    train_loader = DataLoader(train_dataset, params_aux["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, params_aux["batch_size"], shuffle=True)
    train_all_loader = DataLoader(train_all_dataset, params_aux["batch_size"], 
                                  shuffle=True)
    test_loader = DataLoader(test_dataset, params["batch_size"], shuffle=True)
    return train_loader, val_loader, train_all_loader, test_loader

def predict(model: nn.Module, predict_loader: DataLoader):
    """
    Función que dado un modelo y un conjunto de datos (de la clase Dataloader)
    nos devuleve las probabilidades que sea de la clase 0 (electrón) y de la
    clase 1 (fotón) así como la clase verdadera y su energía.

    Parameter
    ---------
    model: Modelo que usamos para predecir
    device: Device donde esta cargado el modelo
    predict_loader: loader de conjunto que se quiere predecir

    Return
    ---------
    probs: matriz donde la primera columna tiene la probabilidad de que el 
    evento sea un electrón y la segunada la probabilidad de que sea fotón
    labels: etiquetas reales de los eventos 
    energy: vector con las energías de los eventos
    """
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    val_size = len(predict_loader.dataset)
    batch_size = predict_loader.batch_size
    probs = np.zeros([val_size, 2])
    labels = np.zeros(val_size)
    energy = np.zeros(val_size)

    # entreamos en el modelo sin modificar sus valores
    with torch.no_grad():
        # iteramos por batchs
        for batch_idx, (inputs, labels_batch, energy_batch) in enumerate(predict_loader):
            inputs_batch = inputs.to(device, dtype=torch.float)
            labels_batch = labels_batch.to(device)
            energy_batch = energy_batch.to(device)
            outputs = model(inputs_batch)
            # TODO si no acaba con softmax no se muy bien como hacerlo
            prob_batch = torch.exp(outputs)
            probs[batch_size * batch_idx:batch_size * (batch_idx + 1), :] = prob_batch.cpu().numpy()
            labels[batch_size * batch_idx:batch_size * (batch_idx + 1)] = labels_batch.cpu().numpy()
            energy[batch_size * batch_idx:batch_size * (batch_idx + 1)] = energy_batch.cpu().numpy()
    return probs, labels, energy

def get_some_data(seed, n, params, e_min, e_max):
    aux = {'seed_': seed}
    fix_seed(aux)
    dataset, _ = get_final_data(params)
    loader = DataLoader(dataset, 1, shuffle=True)
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    img_final = torch.Tensor().to(device)
    energy_final = torch.Tensor().to(device)
    labels_final = torch.Tensor().to(device)
    for inputs, labels_batch, energy_batch in loader:
        inputs_batch = inputs.to(device, dtype=torch.float)
        labels_batch = labels_batch.to(device)
        energy_batch = energy_batch.to(device)
        if (energy_batch > e_min).item() and (energy_batch < e_max).item():
            img_final = torch.concat([img_final, inputs_batch])
            energy_final = torch.concat([energy_final, energy_batch])
            labels_final = torch.concat([labels_final, labels_batch])
            if energy_final.shape[0] >= n:
                labels_final = labels_final.to("cpu").numpy().astype(int)
                energy_final = energy_final.to("cpu").numpy()
                return img_final, labels_final, energy_final

def predict_some_data(inputs, model):
    outputs = model(inputs)
    probs = torch.exp(outputs)
    imgs = inputs.to("cpu").numpy()
    probs = probs.to("cpu").detach().numpy()
    return imgs, probs

def mult_plot(img, y_true, energy, probs, model, n_cols=3):
    fig = plt.figure(figsize=(10, 10))
    img_e, y_true_e, probs_e, y_pred_e, energy_e = class_filter(img, y_true,
                                                                energy, probs,
                                                                0, True)
    print(img_e.shape)
    img_p, y_true_p, probs_p, y_pred_p, energy_p = class_filter(img, y_true,
                                                                energy, probs,
                                                                1, True)
    grid = gridspec.GridSpec(4, n_cols * 2)
    plot_normals(n_cols, fig, 0, img_e, y_true_e, probs_e, y_pred_e, energy_e,
                 img_p, y_true_p, probs_p, y_pred_p, energy_p, grid)
    plot_cam(model, n_cols, 1, fig, img_e, y_true_e, probs_e, y_pred_e,
             energy_e, img_p, y_true_p, probs_p, y_pred_p, energy_p, grid)

    img_e, y_true_e, probs_e, y_pred_e, energy_e = class_filter(img, y_true,
                                                                energy, probs,
                                                                0, False)
    print(img_e.shape)
    img_p, y_true_p, probs_p, y_pred_p, energy_p = class_filter(img, y_true,
                                                                energy, probs,
                                                                1, False)
    plot_normals(n_cols, fig, 2, img_e, y_true_e, probs_e, y_pred_e, energy_e,
                 img_p, y_true_p, probs_p, y_pred_p, energy_p, grid)
    plot_cam(model, n_cols, 3, fig, img_e, y_true_e, probs_e, y_pred_e,
             energy_e, img_p, y_true_p, probs_p, y_pred_p, energy_p, grid)

def plot_normals(n_cols, fig, fila, img_e, y_true_e, probs_e, y_pred_e, energy_e, img_p, y_true_p, probs_p, y_pred_p, energy_p, grid):
    for idx in range(n_cols):
        ax = fig.add_subplot(grid[fila, idx])
        ax.axis("off")
        ax.imshow(img_e[idx, :, :], "gray")
        title = f"true: {y_true_e[idx]}, pred: {y_pred_e[idx]}\n" + \
            f"p. elec {probs_e[idx, 0]:.2f} p. phot {probs_e[idx, 1]:.2f}\n" + \
            f"energy {energy_e[idx]}"
        ax.set_title(title, fontdict={'fontsize': 9, 'fontweight': 'medium'})
        plt.tight_layout()
        ax = fig.add_subplot(grid[fila, idx + 3])
        ax.axis("off")
        ax.imshow(img_p[idx, :, :], "gray")
        title = f"true: {y_true_p[idx]}, pred: {y_pred_p[idx]}\n" + \
            f"p. elec {probs_p[idx, 0]:.2f} p. phot {probs_p[idx, 1]:.2f}\n" + \
            f"energy {energy_p[idx]}"
        ax.set_title(title, fontdict={'fontsize': 9, 'fontweight': 'medium'})
        plt.tight_layout()

def plot_cam(model, n_cols, fila, fig, img_e, y_true_e, probs_e, y_pred_e, energy_e, img_p, y_true_p, probs_p, y_pred_p, energy_p, grid):
    for idx in range(n_cols):
        ax = fig.add_subplot(grid[fila, idx])
        ax.axis("off")
        ax.imshow(img_e[idx, :, :], "gray")
        title = f"true: {y_true_e[idx]}, pred: {y_pred_e[idx]}\n" + \
            f"p. elec {probs_e[idx, 0]:.2f} p. phot {probs_e[idx, 1]:.2f}\n" + \
            f"energy {energy_e[idx]}"
        ax.set_title(title, fontdict={'fontsize': 9, 'fontweight': 'medium'})
        cam = get_cam(img_e[idx, :, :], y_true_e[idx], model)
        ax.imshow(cam)
        plt.tight_layout()
        ax = fig.add_subplot(grid[fila, idx + 3])
        ax.axis("off")
        ax.imshow(img_p[idx, :, :], "gray")
        title = f"true: {y_true_p[idx]}, pred: {y_pred_p[idx]}\n" + \
            f"p. elec {probs_p[idx, 0]:.2f} p. phot {probs_p[idx, 1]:.2f}\n" + \
            f"energy {energy_p[idx]}"
        ax.set_title(title, fontdict={'fontsize': 9, 'fontweight': 'medium'})
        cam = get_cam(img_p[idx, :, :], y_true_p[idx], model)
        ax.imshow(cam)
        plt.tight_layout()

def class_filter(img, y_true, energy, probs, cls, well_classified: bool):
    img_ = img[y_true == cls, :, :]
    y_true_ = y_true[y_true == cls]
    energy_ = energy[y_true == cls]
    probs_ = probs[y_true == cls, :]
    y_pred_ = probs.argmax(axis=1)[y_true == cls]

    if well_classified:
        img_ = img_[y_true_ == y_pred_, 0, :, :]
        energy_ = energy_[y_true_ == y_pred_]
        probs_ = probs_[y_true_ == y_pred_, :]
        y_pred_0 = y_pred_[y_true_ == y_pred_]
        y_true_ = y_true_[y_true_ == y_pred_]
    else:
        img_ = img_[y_true_ != y_pred_, 0, :, :]
        energy_ = energy_[y_true_ != y_pred_]
        probs_ = probs_[y_true_ != y_pred_, :]
        y_pred_0 = y_pred_[y_true_ != y_pred_]
        y_true_ = y_true_[y_true_ != y_pred_]
    return img_, y_true_, probs_, y_pred_0, energy_ 
 
def metrics_calculation(probs, y_true, vervose=True):
    """
    Obtenemos métricas globales dadas las probabilidades obtenidas por el 
    modelo y sus etiquetas verdaderas

    Parameters
    ----------
    probs: Probabilidades de clase 0 y clase 1 obtenidos por el modelo
    y_true: Etiquetas verdaderas de los eventos estudiados
    Return
    ----------
    con_mat: Matriz de confusión
    acc: Accuracy
    f1: Métrica f1
    auc: Métrica auc
    efficiency: Eficiencia
    purity: Pureza
    """
    y_pred = probs.argmax(axis=1)
    # Matriz de confusion
    con_mat = get_con_mat(y_true, y_pred)
    if vervose:
        plot_con_mat(con_mat)
    # Acuracy
    acc = metrics.accuracy_score(y_true, y_pred)
    # F1
    f1 = metrics.f1_score(y_true, y_pred)
    # AUC
    auc = metrics.roc_auc_score(y_true, probs[:, 1])
    # ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true, probs[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    if vervose:
        plot_roc(tpr, fpr, roc_auc)
    # eficiencia y pureza
    efficiency, purity = efficiency_purity(con_mat)
    if vervose:
        print("Acuracy:", round(acc, 4))
        print("F1 Score:", round(f1, 4))
        print("AUC:", round(auc, 4))
        print("Eficiencia:", round(efficiency, 4))
        print("Pureza:", round(purity, 4))
    return con_mat, acc, f1, auc, efficiency, purity

def get_con_mat(y_true, y_pred):
    """
    Fun aux para obtener la mat de confusión dados el y_true y el y_pred 
    """
    con_mat = metrics.confusion_matrix(y_true, y_pred)
    lables_name = ["electron", "photon"]
    con_mat = pd.DataFrame(con_mat, index=lables_name, columns=lables_name)
    return con_mat

def plot_con_mat(con_mat):
    """
    Fun aux para plotear la mat de confusión
    """
    sns.heatmap(con_mat, annot=True, cbar=False)
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")

def plot_roc(tpr, fpr, roc_auc):
    """
    Fun aux para plotear la curva roc
    """
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()

def efficiency_purity(con_mat):
    """
    Función que dada la matriz de confusión obtiene tanto la pureza como la 
    effciencia
    """
    true_elec = con_mat.iloc[0, 0]
    false_photon = con_mat.iloc[0, 1]
    false_elec = con_mat.iloc[1, 0]
    efficiency = true_elec / (true_elec + false_photon)
    purity = true_elec / (false_elec + true_elec)
    return efficiency, purity

def efficiency_purity_energy(probs, y_true, energy, case_of_study,
                             n_bins=50, save: bool=False):
    """
    Plotea la eficiencia y pureza en función de la energía usando diagramas 
    de baras.

    Parameters
    ----------
    probs: probabilades dadas por el modelos 
    y_true: valores verdaderos de los eventos
    energy: energía del evento
    params: diccionario con los parámetros del modelo que se esta estudiando
    n_bins: númeno de barras que tiene el plot
    save: Si se guarda o no la imagen
    """
    y_pred = probs.argmax(axis=1)
    # obnemos lo límites de la energía y hacemos los intervalos usando linspace
    e_max = energy.max()
    e_min = energy.min()
    intervals = np.linspace(e_min, e_max, n_bins)
    # desplazamos para arriba y para bajo para tener los intervalos
    uppers = intervals[1:]
    lowers = intervals[:-1]
    purity_vect = np.zeros(n_bins - 1)
    eff_vect = np.zeros(n_bins - 1)
    counter = 0
    # iteramos en estos intervalos
    for low, up in zip(lowers, uppers):
        # condicion de que estemos dentro del intervalo
        condition = (energy >= low) & (energy < up)
        y_pred_e = y_pred[condition]
        y_true_e = y_true[condition]
        # obtenemos la matriz de confusión dentro de este intervalo de energía
        con_mat_e = get_con_mat(y_true_e, y_pred_e)
        # obtenemos la eff y pureza para ese intervalo de energía
        eff_vect[counter], purity_vect[counter] = efficiency_purity(con_mat_e) 
        counter += 1
    # repetimos dos veces los valores que definen los intervalos de energía
    # menos los los extremos
    e_hist = np.repeat(intervals, 2)[1:-1]
    # repetimos la eficiencia y la pureza para hacer los hist
    eff_hist = np.repeat(eff_vect, 2)
    purity_hist = np.repeat(purity_vect, 2)
    # hacemos el plot
    plt.plot(e_hist, eff_hist, label="efficiency")
    plt.plot(e_hist, purity_hist, label="purity")
    plt.plot(e_hist, eff_hist * purity_hist, label="eff · purity")
    plt.xlabel("Energía")
    plt.legend()
    plt.show()
    if save:
        plt.savefig(f"graficas/eff_purity/{case_of_study}.pdf")

def get_metrics(save: bool, model, train_loader, case_of_study):
    probs, y_true, energy = predict(model, train_loader)
    _ = metrics_calculation(probs, y_true)
    _ = efficiency_purity_energy(probs, y_true, energy, case_of_study,
                                 save=save)

def metric_study(run_id, transform=None, case_of_study="train", save=False):
    params, model = get_model(run_id)
    train_loader, val_loader, _, test_loader = get_data(params, transform)
    if case_of_study == "train":
        get_metrics(save, model, train_loader, case_of_study)
    elif case_of_study == "val":
        get_metrics(save, model, val_loader, case_of_study)
    elif case_of_study == "test":
        get_metrics(save, model, test_loader, case_of_study)
    else:
        raise Exception(f"Not valid case_of_study: {case_of_study}")

def study_foto(run_id, seed, e_min=0, e_max=1000):
    params, model = get_model(run_id)
    img_final, labels_final, energy_final = get_some_data(seed, 200, params, e_min, e_max)
    imgs, probs = predict_some_data(img_final, model)
    mult_plot(imgs, labels_final, energy_final, probs, model)

def remove_hits(x, p):
    x_ = torch.clone(x)
    x_[x_ > (x.max() * p)] = 0
    return x_

def remove_p_hits(x, p):
    x_ = torch.clone(x)
    hits = x_[x_ > 0.00001]
    n_hits = hits.shape[0]
    # TODO crear un vector de longitud n_hits con p * n_hits Falses
    idx = np.random.rand(n_hits) > p
    for i in x_[x_ > 0.00001][idx]:
        x_[x_ == i] = 0
    return x_

def removing_hits_study(run_id, save=False):
    ratios = np.linspace(0., 1, 30)
    purities = []
    params, model = get_model(run_id)
    for p in ratios:
        transform = T.Lambda(lambda x: remove_hits(x, p))
        train_loader, val_loader, _, test_loader = get_data(params, transform)
        probs, y_true, energy = predict(model, val_loader)
        results = metrics_calculation(probs, y_true, False)
        eff = results[4]
        purity = results[5]
        purities.append(purity)
    plt.plot(list(ratios), purities, "o")
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"graficas/resultados/purity_less_hits_{timestamp}.pdf")

def removing_hits_study_total(run_id, save=False):
    ratios = np.linspace(0.1, 1, 5)
    purities = []
    params, model = get_model(run_id)
    for p in ratios:
        transform = T.Lambda(lambda x: remove_p_hits(x, p))
        train_loader, val_loader, _, test_loader = get_data(params, transform)
        probs, y_true, energy = predict(model, val_loader)
        results = metrics_calculation(probs, y_true, False)
        eff = results[4]
        purity = results[5]
        purities.append(purity)
    plt.plot(list(ratios), purities, "o")
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"graficas/resultados/purity_less_hits_total{timestamp}.pdf")
#%%
if __name__ == "__main__":
    # run_id = "259444dccbbc4e5d8e3a3c70f463ae2f"
    # study = "val"
    # metric_study(run_id, study, save=False)
    run_id = "ea7ab94303af439fa9c8a5d364b84a5a"
    # study_foto(run_id, 62)
    removing_hits_study_total(run_id, save=False)

# %%
