#%%
from pickletools import float8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from skimage import color
import seaborn as sns
import pickle

px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

# Leemos los datos
all_events = pd.concat(
    [pd.read_csv("../data/photons_train.csv"),
    pd.read_csv("../data/electrons_train.csv")],
    axis=0
)
# Definimimos los límites de las imágenes
x_min = all_events.hitsX.min() + 5
x_max = all_events.hitsX.max() + 5
y_min = all_events.hitsY.min() + 5
y_max = all_events.hitsY.max() + 5
z_min = all_events.hitsZ.min() + 5
z_max = all_events.hitsZ.max() + 5
# %%
def data_to_array(event, pdg_code, fig, fig_size=(200, 200)):
    x = fig_size[1]
    y = fig_size[0]
    imagen = np.zeros((x, y, 3))
    df = all_events.query(
        f"Event == {event} and PDGcode == {pdg_code}"
    )
    sns.scatterplot(
        data=df,
        x="hitsX",
        y="hitsZ",
        hue="hitsCharge",
        linewidth=0,
        palette="gray",
    )
    plt.xlim(x_min, x_max)
    plt.ylim(z_min, z_max)
    plt.legend("", frameon=False)
    plt.axis("off")
    imagen[:, :, 0] = plot_to_array(fig)
    plt.clf()
    sns.scatterplot(
        data=df,
        x="hitsY",
        y="hitsZ",
        hue="hitsCharge",
        linewidth=0,
        palette="gray",
    )
    plt.xlim(y_min, y_max)
    plt.ylim(z_min, z_max)
    plt.legend("", frameon=False)
    plt.axis("off")
    imagen[:, :, 1] = plot_to_array(fig)
    plt.clf()
    sns.scatterplot(
        data=df,
        x="hitsX",
        y="hitsY",
        hue="hitsCharge",
        linewidth=0,
        palette="gray"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend("", frameon=False)
    plt.axis("off")
    imagen[:, :, 2] = plot_to_array(fig)
    return imagen

# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def plot_to_array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = color.rgb2gray(color.rgba2rgb(im))
    return im

def read_train(fig_size=(200, 200)):
    x = fig_size[1]
    y = fig_size[0]
    fig = plt.figure(figsize=(x * px, y * px))
    print("Reading data: ")
    photons = all_events.query("PDGcode == 22")
    events_photons = np.unique(photons.Event)
    electrons = all_events.query("PDGcode == 11")
    events_electrons = np.unique(electrons.Event)
    n_instances = len(events_electrons) + len(events_photons)
    X_train = np.zeros((n_instances, y, x, 3), dtype=np.float32)
    y_train = np.zeros(n_instances)
    event_id = np.zeros((n_instances, 2))
    y_train[:len(events_photons)] = 0
    y_train[len(events_photons):] = 1
    j = 0
    print("Converting the data: ")
    
    for i in events_photons:
        print(j)
        imagen = data_to_array(i, 22, fig, fig_size)
        X_train[j, :, :, :] = imagen
        event_id[j, 0] = 22
        event_id[j, 1] = i
        j += 1
        if j == 5:
            break
    # for i in events_electrons:
    #     data_to_array(i, 11, fig, fig_size)
    #     X_train[j, :, :, :] = imagen
    #     event_id[j, 0] = 11
    #     event_id[j, 1] = i
    #     j += 1
    
    # with open("train_data.pickle", "wb") as pickle_out:
    #     pickle.dump(
    #         [X_train, y_train, event_id],
    #         pickle_out
    #     )

    return X_train[:10, :, :, :], y_train, event_id

    
# %%
if __name__ == "__main__":
    X_train, y_train, event_id = read_train()
    
# %%
