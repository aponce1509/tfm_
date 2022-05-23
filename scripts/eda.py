#%%
from itertools import count
import matplotlib.pyplot as plt
import pandas as pd
from graficas_pdf import vistas_detector
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size": 15
})
# %%
# Leemos los datos
all_events = pd.concat(
    [pd.read_csv("data/photons_train.csv"),
    pd.read_csv("data/electrons_train.csv")],
    axis=0
)

all_events = all_events.reset_index()
# %%
all_events_grouped = all_events.groupby(["Event", "PDGcode"])
a = all_events.drop(["Energy", "hitsCharge"], axis=1).\
    groupby(["Event", "PDGcode"])
energy = all_events_grouped.max().Energy
ranges = a.max() - a.min()
#%%

print(ranges.describe().drop('index', axis=1).to_latex())
# %% Hacemos los histogramas del rango 
plt.hist(ranges["hitsX"], histtype='step', bins=50, label='x')
plt.hist(ranges["hitsY"], histtype='step', bins=50, label='y')
plt.hist(ranges["hitsZ"], histtype='step', bins=50, label='z')
plt.legend()
plt.title('Distribución de los eventos en el rango')
plt.xlabel('Rango (cm)')
plt.ylabel('Frecuencia')
plt.savefig('latex/img/hist_rango.pdf')
# %%
all_events.describe()
# %% Histográma de las energías
plt.hist(energy, alpha=0.5, bins=20)
plt.title('Distribución de los eventos en energía')
plt.xlabel('Energía, E (GeV)')
plt.ylabel('Frecuencia')
plt.savefig('latex/img/hist_en.pdf')
# %%
vistas_detector(all_events, 1, 11, False)
# %%
# Busqueda de los eventos con pocos hits:
counts = all_events.groupby(["Event", "PDGcode"]).size()
energy = all_events.groupby(["Event", "PDGcode"]).mean().Energy
counts = pd.concat([counts, energy], axis=1).sort_values(0)

# %%
plt.plot(counts['Energy'], counts[0], 'bo', markersize=0.5)
plt.title('Relación entre nº de hits y Energía')
plt.xlabel('Energía, E (GeV)')
plt.ylabel('Nº de hits')
plt.savefig('latex/img/scatter_n_e.pdf')
# %%
all_events.describe()
# %%
import numpy as np
# %%
import numpy as np
ch_max = all_events['hitsCharge'].max() * 1.15



aux_2 = all_events['hitsCharge'] / ch_max
aux_2.hist(histtype='step', bins=20)
# %%

aux_2 = all_events['hitsCharge'] / ch_max
aux_2.hist(histtype='step', bins=20, label='normal')
aux = np.log(all_events['hitsCharge']) / np.log(ch_max)
aux.hist(histtype='step', bins=20, label='log')
#%%
_ = plt.hist(aux, alpha=0.5, bins=20, label='log')
_ = plt.hist(aux_2, alpha=0.5, bins=20, label='Normal')
plt.title('Distribución de la carga por hit')
plt.xlabel('$Q / Q_{max}$')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('latex/img/hist_charge.pdf')
# %%
c = []
a = []
b = []
for i, index in zip(counts[0], counts.index):
    event = index[0]
    pdg_code = index[1]
    for j in range(0, i):
        a.append(event)
        b.append(pdg_code)
        c.append(j)

print(len(a))
print(len(b))
print(len(c))
# %%
columns = {
    'Event': a,
    'PDGcode': b,
    'Time': c
}

df_time = pd.DataFrame(columns)
# %%
df_time = df_time.sort_values(['Event', 'PDGcode', 'Time'])
# %%
df_time = df_time.reset_index()

# %%

all_event_order = all_events.sort_values(['Event', 'PDGcode', 'index'])
#%%
all_event_order = all_event_order.reset_index()
# %%
all_event_order['Time'] = df_time['Time']
#%%
aux = all_event_order.groupby(['Event', 'PDGcode']).max()['Time']
aux = all_event_order.join(aux, on=['Event', 'PDGcode'], lsuffix='0')
# %%
aux['Time'] = aux['Time0'] / aux['Time']
# %%
ch_max = aux['hitsCharge'].max() * 1.15
aux['hitsCharge_t'] = np.log(aux['hitsCharge']) / np.log(ch_max)
# %%
plt.figure(figsize=(7, 4))
_ = plt.hist2d(aux['Time'], aux['hitsCharge_t'], bins=30)
plt.title('Distribución de la carga de los hits a lo largo del tiempo')
plt.colorbar()
plt.xlabel('Tiempo / Tiempo$_{max}$')
plt.ylabel('$\log(Q / Q_{max})$')
plt.savefig('latex/img/hist_charge_time.pdf')
# %%
# start with a square Figure
fig = plt.figure(figsize=(8, 8))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

ax.hist2d(aux['Time'], aux['hitsCharge_t'], bins=20)
# ax_histx.hist(aux['Time'])
ax_histy.hist(aux['hitsCharge_t'])
plt.show()
# %%
plt.hist(aux['hitsCharge_t'])

# %%
sns.jointplot(x=aux['Time'], y=aux['hitsCharge_t'], kind='hex')