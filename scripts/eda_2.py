# %%
%load_ext autoreload
%autoreload 2
from utils_win_cube_copy import *
from graficas_pdf import vistas_detector
# %%

all_events_0 = pd.concat(
    [pd.read_csv("data/photons_train.csv"),
    pd.read_csv("data/electrons_train.csv")],
    axis=0
)

# all_events_0 = all_events.copy()
# %%
min_max, max_charge, range_y_sumed, range_hits_charge_sumed = get_min_max(all_events_0)

# %%
cube_shape_x = 1000
cs = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
all_events = rescale_axis(all_events_0, min_max, max_charge, cs, 'y')


pdg_code = 11
event = 682

df = all_events.query(
    f"Event == {event} and PDGcode == {pdg_code}"
).copy() 
df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
_, img = set_window(
    df,
    projection='color',
    win_shape=(128, 62, 128),
    cube_pool='mean',
    projection_pool='max',
    range_hits_charge_sumed=range_hits_charge_sumed,
    range_y_sumed=range_y_sumed
)

# %%
# img = img.numpy()
plt.figure()
plt.imshow(img.transpose((1, 2, 0)))
plt.title("Proyección de Color")
plt.savefig('latex/img/proyeccion_color.pdf')
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