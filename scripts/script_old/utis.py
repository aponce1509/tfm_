# %%
from matplotlib import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

os.chdir("..")
# %%
def plot_vistas(df, win_shape=(64, 64, 128)):

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, 0])

    sns.scatterplot(
    data=df,
    x="hitsX",
    y="hitsZ",
    hue="hitsCharge",
    linewidth=0,
    palette="flare",
    ax=ax
    )
    if not win_shape == None:
        ax.set_xlim(0, win_shape[0] - 1)
        ax.set_ylim(0, win_shape[2] - 1)
    # ax.legend().set_visible(False)
    # plt.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
    data=df,
    x="hitsY",
    y="hitsZ",
    hue="hitsCharge",
    linewidth=0,
    palette="flare",
    ax=ax
    )
    if not win_shape == None:
        ax.set_xlim(0, win_shape[1] - 1)
        ax.set_ylim(0, win_shape[2] - 1)
    # ax.legend().set_visible(False)
    # plt.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    sns.scatterplot(
    data=df,
    x="hitsX",
    y="hitsY",
    hue="hitsCharge",
    linewidth=0,
    palette="flare",
    ax=ax
    )
    if not win_shape == None:
        ax.set_xlim(0, win_shape[0] - 1)
        ax.set_ylim(0, win_shape[1] - 1)


def get_min_max(data, ids_train=None):
    # selecciono los ids del electro
    if not type(ids_train) == "NoneType":
        ids_electrons = ids_train[ids_train["PDGcode"] == 11]["Event"]
        ids_photons = ids_train[ids_train["PDGcode"] == 22]["Event"]
        data_electrons = data[data["PDGcode"] == 11]
        data_electrons = data_electrons[data_electrons["Event"].\
            isin(ids_electrons)]
        data_photons = data[data["PDGcode"] == 22]
        data_photons = data_photons[data_photons["Event"].isin(ids_photons)]
        data_train = pd.concat([data_photons, data_electrons], axis=0)
    else:
        data_train = data    
    min_max = []
    min_max.append(data_train.hitsX.min() - 5)
    min_max.append(data_train.hitsX.max() + 5)
    min_max.append(data_train.hitsY.min() - 5)
    min_max.append(data_train.hitsY.max() + 5)
    min_max.append(data_train.hitsZ.min() - 5)
    min_max.append(data_train.hitsZ.max() + 5)
    max_charge = data_train.hitsCharge.max()
    return min_max, max_charge

# Esta función modifica data de manera global no se si es lo correcto
def rescale_axis(data, mins_maxs, max_charge, cube_shape=(1000, 1000, 1000)):
    names = ["hitsX", "hitsY", "hitsZ", "hitsCharge"]
    hits_idx = [data.columns.get_loc(i) for i in names]
    data.iloc[:, hits_idx[0]] = round(((data["hitsX"] - mins_maxs[0]) / 
                     (mins_maxs[1] - mins_maxs[0])) * cube_shape[0], 0).\
                         astype(int)
    data.iloc[:, hits_idx[1]] = round(((data["hitsY"] - mins_maxs[2]) / 
                     (mins_maxs[3] - mins_maxs[2])) * cube_shape[1], 0).\
                         astype(int)
    data.iloc[:, hits_idx[2]] = round(((data["hitsZ"] - mins_maxs[4]) / 
                     (mins_maxs[5] - mins_maxs[4])) * cube_shape[2], 0).\
                         astype(int)
    data.iloc[:, hits_idx[2]] = data.iloc[:, hits_idx[2]] / max_charge
    return data

def set_window(
        df, win_shape=(64, 64, 128), projection="3d", cube_pool="mean",
        projection_pool="max"
    ):
    names = ["hitsX", "hitsY", "hitsZ"]
    hits_idx = [df.columns.get_loc(i) for i in names]
    # eje x
    # vemos la dirección de drift
    ################# HACER MEDIA DE VARIOS VALORES

    new_x = df["hitsX"] - df["hitsX"].iloc[0]
    n_hits = new_x.shape[0]
    positve_ratio = (new_x > 0).sum() / n_hits
    if positve_ratio > 0.5:
        x_dir = "positive"
    else:
        x_dir = "negative"
    # hacemos un pool para los puntos que caigan en los mismo pixeles del 
    # bloque completo, es decir puntos muy cercanos
    df_win = df.copy()
    if cube_pool == "mean":
        df_win = df_win.groupby(["hitsX", "hitsY", "hitsZ"]).mean().\
            reset_index()
    elif cube_pool == "max":
        df_win = df_win.groupby(["hitsX", "hitsY", "hitsZ"]).max().\
            reset_index()
    else:
        raise Exception("Cube pool name not valid")
    new_x = df_win["hitsX"] - df["hitsX"].iloc[0]
    # "colocamos" la ventana modificando los indices
    if n_hits > 30:
        if x_dir == "positive":
            df_win.iloc[:, hits_idx[0]] = new_x
        elif x_dir == "negative":
            df_win.iloc[:, hits_idx[0]] = new_x + win_shape[0] - 15
    else:
        df_win.iloc[:, hits_idx[0]] = new_x + int(win_shape[0] / 2)
    df_win.iloc[:, hits_idx[1]] = df_win["hitsY"] - df["hitsY"].iloc[0] +\
        int(win_shape[1] / 2)
    df_win.iloc[:, hits_idx[1]] = win_shape[1] - df_win.iloc[:, hits_idx[1]] 
    df_win.iloc[:, hits_idx[2]] = df_win["hitsZ"] - df_win["hitsZ"].min()

    if projection == "3d":
        df_win = df_win[
            np.logical_and(
                df_win["hitsX"] < win_shape[0],
                df_win["hitsX"] >= 0
            )
        ]
        df_win = df_win[
            np.logical_and(
                df_win["hitsY"] < win_shape[1],
                df_win["hitsY"] >= 0
            )
        ]
        df_win = df_win[
            np.logical_and(
                df_win["hitsZ"] < win_shape[2],
                df_win["hitsZ"] >= 0
            )
        ]
        image = np.zeros(win_shape)
        for index, row in df_win.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsY"])
            z = int(row["hitsZ"])
            image[x, y, z] = row["hitsCharge"]
            return df_win, image
    elif projection == "z":
        df_win = df_win.drop(["hitsZ"], axis=1)
        if projection_pool == "max":
            df_win = df_win.groupby(["hitsX", "hitsY"]).max().reset_index()
        elif projection_pool == "mean":
            df_win = df_win.groupby(["hitsX", "hitsY"]).mean().reset_index()
        df_win = df_win[
            np.logical_and(
                df_win["hitsX"] < win_shape[0],
                df_win["hitsX"] >= 0
            )
        ]
        df_win = df_win[
            np.logical_and(
                df_win["hitsY"] < win_shape[1],
                df_win["hitsY"] >= 0
            )
        ]
        image = np.zeros((win_shape[0], win_shape[1]))
        for index, row in df_win.iterrows():
            y = int(row["hitsX"])
            x = int(row["hitsY"])
            image[x, y] = row["hitsCharge"]
        image = image.reshape((1, win_shape[0], win_shape[1]))
        return df_win, image
    else:
        raise Exception("Projection name not valid")


# %%

class Cascadas(Dataset):
    def __init__(
        self, seed_=123, train: bool=True, validation: bool=True,
        transform=None, cube_shape=(2000, 2000, 2500), win_shape=(62, 62, 128),
        projection="z", cube_pool="mean", projection_pool="max"
    ) -> None:
        super().__init__()
        self.win_shape = win_shape
        self.cube_shape = cube_shape
        self.projection = projection
        self.cube_pool = cube_pool
        self.projection_pool = projection_pool
        if validation:  # validation == True si no estamos con el test
            # cargamos los datos
            self.all_events = pd.concat(
                [pd.read_csv("data/photons_train.csv"),
                pd.read_csv("data/electrons_train.csv")],
                axis=0
            )           
            # cogemos los eventos de train y validacion y su etiqueta
            ids_train, ids_test = train_test_split(
                self.all_events[["Event", "PDGcode"]].drop_duplicates(),
                random_state=seed_, test_size=0.1
            )
            # calculamos el máximo y el minimo de los ejes para train
            mins_maxs, max_charge = get_min_max(self.all_events, ids_train)
            # reescalamos todos pero usando los limites dados por train
            self.all_events = rescale_axis(
                self.all_events, mins_maxs, max_charge, cube_shape
            )
            if train:
                self.events = ids_train["Event"]
                self.labels = ids_train["PDGcode"]
            else:
                self.events = ids_test["Event"]
                self.labels = ids_test["PDGcode"]
        # Si no estamos en validación
        else:
        # Cargo train siempre porque tengo que calcular sus límites
            self.all_events = pd.concat(
                [pd.read_csv("data/photons_train.csv"),
                pd.read_csv("data/electrons_train.csv")],
                axis=0
            )
            mins_maxs, max_charge = get_min_max(self.all_events)
            if train:
                ids = self.all_events[["Event", "PDGcode"]].drop_duplicates()
                self.all_events = rescale_axis(
                    self.all_events, mins_maxs, max_charge, cube_shape
                )
                self.events = ids["Event"]
                self.labels = ids["PDGcode"]
            else:
                self.all_events = pd.concat(
                    [pd.read_csv("data/photons_test.csv"),
                    pd.read_csv("data/electrons_test.csv")],
                    axis=0
                )
                self.all_events = rescale_axis(
                    self.all_events, mins_maxs, max_charge, self.cube_shape
                )
                ids = self.all_events[["Event", "PDGcode"]].drop_duplicates()
                self.events = ids["Event"]
                self.labels = ids["PDGcode"]
        self.transform = transform
        
    def __len__(self):
         return len(self.labels)
    
    def __getitem__(self, idx):
        event = self.events.iloc[idx]
        label = self.labels.iloc[idx]
        df = self.all_events.query(
            f"Event == {event} and PDGcode == {label}"
        ).copy()   
        df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
        _, img = set_window(
            df, projection=self.projection, win_shape=self.win_shape,
            cube_pool=self.cube_pool, projection_pool=self.projection_pool
        )
        self.colnames = df.columns.values
        df = df.to_numpy()
        # IMAGEN RESHAPE
        # if self.transform is not None:
        #     img = self.transform(img)
        if label == 11:
            label = 0
        elif label == 22:
            label = 1
        return img, label

    def plot(self, idx):
        img, _ = self[idx]
        print(self.events.iloc[idx], self.labels.iloc[idx])
        # colnames = ["hitsX", "hitsY", "hitsZ", "hitsCharge"]
        # df_aux = pd.DataFrame(df, columns=self.colnames)
        # # el evento dento del bloque definido
        # plot_vistas(df_aux, None)
        # plt.show()
        # el evento en la venta
        if not self.projection == "3d":
            plt.figure()
            plt.imshow(img[0, :, :], cmap="gray")
            plt.show()

# %%
if __name__ == "__main__":
    data = Cascadas()
    img, label = data[7]
    data.plot(7)
# %%
