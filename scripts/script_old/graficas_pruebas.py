# %% Modulos

from email.mime import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
import io
from skimage import color

# %% load data
all_events = pd.concat(
    [pd.read_csv("data/photons.csv"), pd.read_csv("data/electrons.csv")],
    axis=0
)
# %%
x_min = all_events.hitsX.min() + 5
x_max = all_events.hitsX.max() + 5
y_min = all_events.hitsY.min() + 5
y_max = all_events.hitsY.max() + 5
z_min = all_events.hitsZ.min() + 5
z_max = all_events.hitsZ.max() + 5

# %%
def vistas_detector(event, pdg_code, save=True):
    df = all_events.query(
        f"Event == {event} and PDGcode == {pdg_code}"
    )
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
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(z_min, z_max)
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
    # ax.set_xlim(y_min, y_max)
    # ax.set_ylim(z_min, z_max)
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
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.legend().set_visible(False)
    # plt.axis("off")
    if pdg_code == 22:
        fig.suptitle("Photon", fontsize=14)
    elif pdg_code == 11:
        fig.suptitle("Electron", fontsize=14)
    if save:
        plt.savefig(f"graficas/{event}_{pdg_code}_vistas.pdf")

def plot3d(event, pdg_code, save=True):
    df = all_events.query(
    f"Event == {event} and PDGcode == {pdg_code}"
    )
    fig = plt.figure(figsize=(16,11))
    ax = fig.add_subplot(111, projection='3d')
    cmap = ListedColormap(sns.color_palette("flare").as_hex())
    sc = ax.scatter(
        xs=df.hitsX,
        ys=df.hitsY,
        zs=df.hitsZ,
        c=df.hitsCharge,
        cmap=cmap
    )
    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(
        xs=df.hitsX,
        ys=df.hitsY,
        zs=0,
    )
    ax.scatter(
        xs=df.hitsX,
        ys=ax.get_ylim()[1],
        zs=df.hitsZ
    )
    if pdg_code == 22:
        plt.title("photon")
    elif pdg_code == 11:
        plt.title("electron")
    if save:
        plt.savefig(f"graficas/{event}_{pdg_code}_3d.pdf")

def vistas_y_3d(event, pdg_code, save=True):
    vistas_detector(event, pdg_code, save)
    plot3d(event, pdg_code, save) 

def vistas_separadas_as_jpg(event, pdg_code):
    path = "../graficas/vistas_separadas/" 
    df = all_events.query(
        f"Event == {event} and PDGcode == {pdg_code}"
    )

    fig = plt.figure(figsize=(200 * px, 200 * px))
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
    plt.savefig(path + f"{event}_{pdg_code}_vistas_xz.png")
    fig = plt.figure(figsize=(200 * px, 200 * px))
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
    plt.savefig(path + f"{event}_{pdg_code}_vistas_yz.png")
    fig = plt.figure(figsize=(200 * px, 200 * px))
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
    plt.savefig(path + f"{event}_{pdg_code}_vistas_xy.png")

def vistas_juntas_as_jpg(event, pdg_code):
    df = all_events.query(
        f"Event == {event} and PDGcode == {pdg_code}"
    )
    fig = plt.figure(figsize=(200 * px, 200 * px))
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, 0])

    sns.scatterplot(
        data=df,
        x="hitsX",
        y="hitsZ",
        hue="hitsCharge",
        linewidth=0,
        palette="gray",
        ax=ax
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.legend().set_visible(False)
    plt.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
        data=df,
        x="hitsY",
        y="hitsZ",
        hue="hitsCharge",
        linewidth=0,
        palette="gray",
        ax=ax
    )
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)
    ax.legend().set_visible(False)
    plt.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    sns.scatterplot(
        data=df,
        x="hitsX",
        y="hitsY",
        hue="hitsCharge",
        linewidth=0,
        palette="gray",
        ax=ax
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend().set_visible(False)
    plt.axis("off")
    path = "../graficas/vistas_juntas/" 
    plt.savefig(path + f"{event}_{pdg_code}_vistas.png")


# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def plot_to_array():
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = color.rgb2gray(color.rgba2rgb(im))
    return im

def data_to_array(event, pdg_code):
    imagen = np.zeros((200, 200, 3))
    df = all_events.query(
        f"Event == {event} and PDGcode == {pdg_code}"
    )
    fig = plt.figure(figsize=(200 * px, 200 * px))
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
    imagen[:, :, 0] = plot_to_array()
    fig = plt.figure(figsize=(200 * px, 200 * px))
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
    imagen[:, :, 1] = plot_to_array()
    fig = plt.figure(figsize=(200 * px, 200 * px))
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
    imagen[:, :, 2] = plot_to_array()

def data_to_img_vista_juntas():
    photons = all_events.query("PDGcode == 22")
    events = np.unique(photons.Event)
    for i in events:
        vistas_juntas_as_jpg(i, 22)
    electrons = all_events.query("PDGcode == 11")
    events = np.unique(electrons.Event)
    for i in events:
        vistas_juntas_as_jpg(i, 11)

def data_to_img_vista_separadas():
    photons = all_events.query("PDGcode == 22")
    events = np.unique(photons.Event)
    for i in events:
        vistas_separadas_as_jpg(i, 22)
    electrons = all_events.query("PDGcode == 11")
    events = np.unique(electrons.Event)
    for i in events:
        vistas_separadas_as_jpg(i, 11)
if __name__ == "__main__":
    # vistas_y_3d(1, 22)
    # vistas_y_3d(1, 22)
    # vistas_y_3d(0, 11)
    # vistas_y_3d(0, 22)
    # vistas_y_3d(1, 11)
    vistas_y_3d(2, 11)
    # data_to_img_vista_juntas()