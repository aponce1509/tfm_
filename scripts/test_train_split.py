import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# cargamos datos
print("Fotones: ")
photons = pd.read_csv("data/photons.csv")
# obtenemos los eventos únicos
events_photons = np.unique(photons.Event)
# dividimos en train y test
train_evn, test_evn = train_test_split(
    events_photons,
    test_size=0.1,
    random_state=123)
# nos quedamos con los fotones de train
train_photons = photons[photons["Event"].isin(train_evn)]
train_photons.to_csv("data/photons_train.csv", index=False)
# nos quedamos con los fotones de test
test_photons = photons[photons["Event"].isin(test_evn)]
test_photons.to_csv("data/photons_test.csv", index=False)
# Cargamos datos 
print("Electrones: ")
electrons = pd.read_csv("data/electrons.csv")
# obtenemos los eventos únicos
events_elecrtons = np.unique(electrons.Event)
# dividimos en train y test
train_evn, test_evn = train_test_split(
    events_elecrtons,
    test_size=0.1,
    random_state=123)
# nos quedamos con los electrones de train
train_electrons = electrons[electrons["Event"].isin(train_evn)]
train_electrons.to_csv("data/electrons_train.csv", index=False)
# nos quedamos con los electrones  de test
test_electrons = electrons[electrons["Event"].isin(test_evn)]
test_electrons.to_csv("data/electrons_test.csv", index=False)