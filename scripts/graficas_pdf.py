# %% Modulos

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

# %%
def vistas_detector(all_events, event, pdg_code, save=True):
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
    # ax.legend().set_visible(False)
    # plt.axis("off")
    if pdg_code == 22:
        fig.suptitle("Photon", fontsize=14)
    elif pdg_code == 11:
        fig.suptitle("Electron", fontsize=14)
    if save:
        plt.savefig(f"graficas/{event}_{pdg_code}_vistas.pdf")

def plot3d(all_events, event, pdg_code, save=True):
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
        alpha=0.8,
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
        color="#2b002b"
    )
    ax.scatter(
        xs=df.hitsX,
        ys=ax.get_ylim()[1],
        zs=df.hitsZ,
        color="#3a005c"
    )
    if pdg_code == 22:
        plt.title("photon")
    elif pdg_code == 11:
        plt.title("electron")
    if save:
        plt.savefig(f"graficas/{event}_{pdg_code}_3d.pdf")

def vistas_y_3d(all_events, event, pdg_code, save=True):
    vistas_detector(all_events, event, pdg_code, save)
    plot3d(all_events, event, pdg_code, save) 

if __name__ == "__main__":
    all_events = pd.concat(
        [pd.read_csv("data/photons.csv"), pd.read_csv("data/electrons.csv")],
        axis=0
    )
    vistas_y_3d(all_events, 2101, 11)
