# %%
from msilib.schema import Error
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np

os.chdir("..")
# %%
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
        data = pd.concat([data_photons, data_electrons], axis=0)
    min_max = []
    min_max.append(data.hitsX.min() - 5)
    min_max.append(data.hitsX.max() + 5)
    min_max.append(data.hitsY.min() - 5)
    min_max.append(data.hitsY.max() + 5)
    min_max.append(data.hitsZ.min() - 5)
    min_max.append(data.hitsZ.max() + 5)
    return min_max

def rescale_axis(data, mins_maxs, cube_shape=(1000, 1000, 1000)):
    names = ["hitsX", "hitsY", "hitsZ"]
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
    return data

def set_window(data, win_shape=(64, 64, 128), pool="mean", proyection="3d"):
    names = ["hitsX", "hitsY", "hitsZ"]
    hits_idx = [data.columns.get_loc(i) for i in names]
    # eje x
    # vemos la dirección de drift
    x_diff = data.hitsX.iloc[0] - data.hitsX.iloc[50]
    print(x_diff)
    if x_diff > 0:
        x_dir = "negative"
    else:
        x_dir = "positive"
    # "colocamos" la ventana
    if x_dir == "positive":
        data.iloc[:, hits_idx[0]] = data["hitsX"] - data["hitsX"].min()
    elif x_dir == "negative":
        data.iloc[:, hits_idx[0]] = data["hitsX"] - data["hitsX"].max() +\
            win_shape[0] - 1
        print(win_shape[0])
    data.iloc[:, hits_idx[1]] = data["hitsY"] - data["hitsY"].iloc[0] +\
        int(win_shape[1] / 2)
    data.iloc[:, hits_idx[2]] = data["hitsZ"] - data["hitsZ"].min()
    if pool == "mean":
        data = data.groupby(["hitsX", "hitsY", "hitsZ"]).mean().reset_index()
    else:
        raise Exception("Pool name not valid")
    if proyection == "3d":
        data = data[
            np.logical_and(
                data["hitsX"] < win_shape[0],
                data["hitsX"] >= 0
            )
        ]
        data = data[
            np.logical_and(
                data["hitsY"] < win_shape[1],
                data["hitsY"] >= 0
            )
        ]
        data = data[
            np.logical_and(
                data["hitsZ"] < win_shape[2],
                data["hitsZ"] >= 0
            )
        ]
        image = np.zeros(win_shape)
        for index, row in data.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsY"])
            z = int(row["hitsZ"])
            image[x, y, z] = row["hitsCharge"]
    elif proyection == "z":
        data.drop(["hitsZ"], axis=1, inplace=True)
        data = data.groupby(["hitsX", "hitsY"]).max().reset_index()
        data = data[
            np.logical_and(
                data["hitsX"] < win_shape[0],
                data["hitsX"] >= 0
            )
        ]
        data = data[
            np.logical_and(
                data["hitsY"] < win_shape[1],
                data["hitsY"] >= 0
            )
        ]
        image = np.zeros((win_shape[0], win_shape[1]))
        for index, row in data.iterrows():
            y = int(row["hitsX"])
            x = int(row["hitsY"])
            image[x, y] = row["hitsCharge"]
    else:
        raise Exception("Projection name not valid")

    return data, image



# %%
    

class Cascadas(Dataset):
    def __init__(
        self, seed_=123, train: bool=True, validation: bool=True,
        transform=None, cube_shape=(2000, 2000, 2000)
    ) -> None:
        super().__init__()

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
            mins_maxs = get_min_max(self.all_events, ids_train)
            # reescalamos todos pero usando los limites dados por train
            self.all_events = rescale_axis(
                self.all_events, mins_maxs, cube_shape
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
            mins_maxs = get_min_max(self.all_events)
            if train:
                ids = self.all_events[["Event", "PDGcode"]].drop_duplicates()
                self.all_events = rescale_axis(
                    self.all_events, mins_maxs, cube_shape
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
                    self.all_events, mins_maxs, cube_shape
                )
                ids = self.all_events[["Event", "PDGcode"]].drop_duplicates()
                self.events = ids["Event"]
                self.labels = ids["PDGcode"]
        self.transform = transform
        # Aplicamos la transformación
    def __len__(self):
         return len(self.labels)
    def __getitem__(self, idx):
        event = self.events.iloc[idx]
        label = self.labels.iloc[idx]
        df = self.all_events.query(
            f"Event == {event} and PDGcode == {label}"
        )
        df.drop(["Event", "PDGcode", "Energy"], axis=1, inplace=True)
        return df, label, event

data = Cascadas()
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

df, label, event = data[7]
df_9, image = set_window(df)
df2, image = set_window(df, proyection="z")

fig = plt.figure(figsize=(16,11))
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
ax.set_xlim(0, 63)
ax.set_ylim(0, 127)
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
ax.set_xlim(0, 63)
ax.set_ylim(0, 127)
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
ax.set_xlim(0, 63)
ax.set_ylim(0, 63)
# %%
plt.imshow(image, cmap="gray")