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
cube_shape_x = 1500
cs = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
all_events = rescale_axis(all_events_0, min_max, max_charge, cs, 'y')


# %%
pdg_code = 22
event = 25

df = all_events.query(
    f"Event == {event} and PDGcode == {pdg_code}"
).copy() 
df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
_, img = set_window(
    df,
    projection='y',
    win_shape=(244, 62, 244),
    cube_pool='mean',
    projection_pool='max',
    range_hits_charge_sumed=None,
    range_y_sumed=None
)


# img = img.numpy()
plt.figure()
plt.imshow(img[0, :, :], cmap="gray")
plt.show()
# %%

vistas_detector(all_events_0, event, pdg_code)
# %%
