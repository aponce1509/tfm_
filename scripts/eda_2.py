# %%
%load_ext autoreload
%autoreload 2
from utils_win_cube_copy import *
from graficas_pdf import vistas_detector
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size": 15
})
# %%

all_events_0 = pd.concat(
    [pd.read_csv("data/photons_train.csv"),
    pd.read_csv("data/electrons_train.csv")],
    axis=0
)
# %% Gráfica tal cual
event_1 = 50
event_2 = 50
pdg_code = 22

df = all_events_0.query(
    f"Event == {event_1} and PDGcode == {11}"
)
fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(1, 2)
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
ax.set_title('Electrón')

pdg_code = 22
df = all_events_0.query(
    f"Event == {event_2} and PDGcode == {22}"
)
ax = fig.add_subplot(gs[0, 1])

sns.scatterplot(
    data=df,
    x="hitsX",
    y="hitsZ",
    hue="hitsCharge",
    linewidth=0,
    palette="flare",
    ax=ax
)
ax.set_title('Fotón')
plt.savefig('latex/img/comparación.pdf')

# %% Gráfica tal cual 3d
event_1 = 50
pdg_code = 11

df = all_events_0.query(
    f"Event == {event_1} and PDGcode == {pdg_code}"
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
# ax.set_title('Electrón')
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
# if pdg_code == 22:
#     fig.suptitle("Photon", fontsize=14)
# elif pdg_code == 11:
#     fig.title("Electron", fontsize=14)

# plt.savefig('latex/img/vistas_3.pdf')

# %% 3d

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
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
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
plt.savefig('latex/img/3d_3.pdf')

# all_events_0 = all_events.copy()
# %%
min_max, max_charge, range_y_sumed, range_hits_charge_sumed = get_min_max(all_events_0)

# %%
cube_shape_x = 500
cs = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
all_events = rescale_axis(all_events_0, min_max, max_charge, cs, 'y')

# %%
pdg_code = 11
event = 50

df = all_events.query(
    f"Event == {event} and PDGcode == {pdg_code}"
).copy() 
df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
a, img = set_window(
    df,
    projection='3d',
    win_shape=(128, 128, 128),
    cube_pool='mean',
    projection_pool='max',
    range_hits_charge_sumed=range_hits_charge_sumed,
    range_y_sumed=range_y_sumed
)

# %%
import numpy as np
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
cm = plt.cm.get_cmap()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
# data = np.random.random(size=(3, 3, 3))
z, y, x = img[0, :, :, :].nonzero()
charge = img[0, z, y, x]
plt.title('Representación tridimensional')
# plt.colorbar()
sc = ax.scatter(x, y, z, c=charge, alpha=1)

cbar = plt.colorbar(sc, fraction=0.041, pad=0.1)
cbar.set_label('Carga normalizada')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('latex/img/rep_3d.pdf')
plt.show()

# %%
# img = img.numpy()
plt.figure()
plt.imshow(img.transpose((1, 2, 0)))
plt.imshow(img[0, :, :], 'gray')
plt.title("Proyección de Color")
# plt.savefig('latex/img/proyeccion_color.pdf')
# %%
# (22) 68,
# (11) 86 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size": 15
})
# %%
plt.figure(figsize=(8, 4))
df_ = all_events_0.query(
    f"Event == {event} and PDGcode == {pdg_code}"
).copy() 
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=df_,
    x="hitsX",
    y="hitsZ",
    hue="hitsCharge",
    linewidth=0,
    palette="flare",
    # ax=ax
)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(img[0, :, :], cmap="gray")
plt.title('Procesado')
plt.savefig('latex/img/ambas_pro.pdf')


# plt.savefig('latex/img/proyeccion_68_22.pdf')

# %%
# img_1 = img.copy()  86 11
# img_2 = img.copy()  68 22
plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(img_1[0, :, :], cmap="gray")
plt.axis('off')
plt.title('Fotón')
plt.subplot(2, 4, 2)
plt.imshow(img_2[0, :, :], cmap="gray")
plt.axis('off')
plt.title('Fotón')
plt.subplot(2, 4, 3)
plt.imshow(img_3[0, :, :], cmap="gray")
plt.axis('off')
plt.title('Fotón')
plt.subplot(2, 4, 4)
plt.imshow(img_4[0, :, :], cmap="gray")
plt.axis('off')
plt.title('Fotón')
plt.subplot(2, 4, 5)
plt.imshow(img_5[0, :, :], cmap="gray")
plt.axis('off')
plt.title('Electrón')
plt.subplot(2, 4, 6)
plt.imshow(img_6[0, :, :], cmap="gray")
plt.axis('off')
plt.title('Electrón')
plt.subplot(2, 4, 7)
plt.imshow(img_7[0, :, :], cmap="gray")
plt.title('Electrón')
plt.axis('off')
plt.subplot(2, 4, 8)
plt.imshow(img_8[0, :, :], cmap="gray")
plt.title('Electrón')
plt.axis('off')
plt.savefig('latex/img/varias_proyecciones.pdf')
# %%
plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.imshow(img_1[0, :, :], cmap="gray")
plt.axis('off')
plt.title('250')
plt.subplot(1, 5, 2)
plt.imshow(img_2[0, :, :], cmap="gray")
plt.axis('off')
plt.title('500')
plt.subplot(1, 5, 3)
plt.imshow(img_3[0, :, :], cmap="gray")
plt.axis('off')
plt.title('1000')
plt.subplot(1, 5, 4)
plt.imshow(img_4[0, :, :], cmap="gray")
plt.axis('off')
plt.title('1500')
plt.subplot(1, 5, 5)
plt.imshow(img_5[0, :, :], cmap="gray")
plt.axis('off')
plt.title('2000')
plt.savefig('latex/img/varias_resoluciones.pdf')