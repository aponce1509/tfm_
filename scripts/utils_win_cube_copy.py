# %%
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
import torch
import pickle
from paths import IMG_PATH
from graficas_pdf import vistas_detector
from matplotlib.colors import ListedColormap

def plot_vistas(df, win_shape: tuple=(64, 64, 128)) -> None:
    """
    Dibujo de las vistas de un evento. Pueden ser con las vistas 
    Parameters
    --------
    df: DataFrame con el evento 
    win_shape: Tupla con el número de índices. Si es None dibuja las vistas 
    completas

    Return
    --------
    None
    """
    
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    # plot vista x-z
    sns.scatterplot(
        data=df,
        x="hitsX",
        y="hitsZ",
        hue="hitsCharge",
        linewidth=0,
        palette="flare",
        ax=ax
    )
    # si fijamos venta, ponemos límites en los ejes y es necesario invertir 
    # los ejes y e z ya que están invertidos para que se dibujen bien como
    # matrices 
    if not win_shape == None:
        ax.set_xlim(0, win_shape[0] - 1)
        ax.set_ylim(0, win_shape[2] - 1)
        ax.invert_yaxis()

    ax = fig.add_subplot(gs[0, 1])
    # plot vista y-z
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
        ax.invert_yaxis()
        ax.invert_xaxis()
    ax = fig.add_subplot(gs[1, 0])
    # plot vista x-y
    sns.scatterplot(
        data=df,
        x="hitsX",
        y="hitsY",
        hue="hitsCharge",
        linewidth=0,
        palette="gray",
        ax=ax
    )
    if not win_shape == None:
        ax.set_xlim(0, win_shape[0] - 1)
        ax.set_ylim(0, win_shape[1] - 1)
        ax.invert_yaxis()

def get_min_max(data: pd.DataFrame, ids_train=None):
    """
    Obtiene los mínimos, máximos de los 3 ejes espaciales y el máximo de la
    carga de todos los eventos dados. Los obtenemos para normalizar y hacer
    un cambio en las unidades de los ejes.
    Parameters
    --------
    data: DataFrame con todos los eventos donde buscar los límites.
    ids_train: ids de los eventos que son de train. 
    None si el DataFrame data tiene solo eventos de train
    Return
    --------
    Tuple: (min_max, max_charge)
    min_max: array con los min y max de los 3 ejes.
    max_charge: mayor valor de hitsCharge

    """
    NoneType = type(None)
    if not type(ids_train) == NoneType:
        # separamos en electrones y fotones
        ids_electrons = ids_train[ids_train["PDGcode"] == 11]["Event"]
        ids_photons = ids_train[ids_train["PDGcode"] == 22]["Event"]
        # dataframe de los electones de train
        data_electrons = data[data["PDGcode"] == 11]
        data_electrons = data_electrons[data_electrons["Event"].\
            isin(ids_electrons)]
        # dataframe de los fotones de train
        data_photons = data[data["PDGcode"] == 22]
        data_photons = data_photons[data_photons["Event"].isin(ids_photons)]
        # juntamos
        data_train = pd.concat([data_photons, data_electrons], axis=0)
    else:
        data_train = data.copy()
    # obtenemos los máximos y mínimos
    y_min_by_event = data_train.groupby('Event')['hitsY'].min().rename('y_min')
    data_train = data_train.join(y_min_by_event, on='Event')
    data_train['hitsY_mod'] = data_train['hitsY'] - data_train['y_min']

    df_aux = data_train.groupby(["Event", "hitsX", "hitsZ"], as_index = False) \
        .sum() \
        .reset_index()
    y_sum_min = df_aux.hitsY_mod.min() - 10
    y_sum_max = df_aux.hitsY_mod.max() + 10
    charge_sum_min = np.log(df_aux.hitsCharge.min() * 0.85)
    charge_sum_max = np.log(df_aux.hitsCharge.max() * 1.15)
    range_y_sumed = [y_sum_min, y_sum_max]
    range_hits_charge_sumed = [charge_sum_min, charge_sum_max]
    min_max = []
    min_max.append(data_train.hitsX.min() - 5)
    min_max.append(data_train.hitsX.max() + 5)
    min_max.append(data_train.hitsY.min() - 5)
    min_max.append(data_train.hitsY.max() + 5)
    min_max.append(data_train.hitsZ.min() - 5)
    min_max.append(data_train.hitsZ.max() + 5)
    max_charge = data_train.hitsCharge.max() * 1.15
    return min_max, max_charge, range_y_sumed, range_hits_charge_sumed

def rescale_axis(data, mins_maxs, max_charge, cube_shape=(2000, 2000, 2500),
                 projection=None, log_trans=False):
    """
    Rescalado de los ejes espaciales y normalización de la carga. El reesclado
    espacial se hace para que la posición de los eventos estén dados por valores
    eneteros entre cero y valor dado por cube_shape. De esta forma se tienen 
    directamente índices para colocar los hits en una matriz. 

    Parameters
    --------
    data: DataFrame con todos los eventos.
    mins_maxs: array con los mínimos y máximos de los 3 ejes (de train).
    max_charge: Valor máximo de la carga (de train)
    cube_shape: Dimensiones del cubo final que representa el volumen del 
    detector.

    Return
    --------
    data: DataFrame con los ejes rescalados
    """
    # Nombres de las columnas a escalar
    data_copy = data.copy()
    names = ["hitsX", "hitsY", "hitsZ", "hitsCharge"]
    # Obtenemos sus ids
    hits_idx = [data_copy.columns.get_loc(i) for i in names]
    # Cambio del eje X
    data_copy.iloc[:, hits_idx[0]] = round(((data_copy["hitsX"] - mins_maxs[0]) / 
                     (mins_maxs[1] - mins_maxs[0])) * cube_shape[0], 0).\
                         astype(int)
    # Cambio del eje Y si no estamos en la projección de color
    # TODO no se si es correcto que no cambie la dimensión y esta en otras 
    # unidades
    if not projection == "color": 
        data_copy.iloc[:, hits_idx[1]] = round(((data_copy["hitsY"] - mins_maxs[2]) / 
                         (mins_maxs[3] - mins_maxs[2])) * cube_shape[1], 0).\
                             astype(int)
    # Cambio del eje Z
    data_copy.iloc[:, hits_idx[2]] = round(((data_copy["hitsZ"] - mins_maxs[4]) / 
                     (mins_maxs[5] - mins_maxs[4])) * cube_shape[2], 0).\
                         astype(int)
    # Cambio de la charga
    if not projection == "color": 
        # En caso que haya valores mayores de la carga máxima (puede darse en 
        # validación) se fija al valor máximo de forma que la carga este acotada
        data_copy.iloc[data_copy["hitsCharge"] >= max_charge, hits_idx[3]] = max_charge 
        data_copy.iloc[data_copy["hitsCharge"] <= 1, hits_idx[3]] = 1
        if log_trans:
            data_copy.iloc[:, hits_idx[3]] = np.log(data_copy.iloc[:, hits_idx[3]])
            data_copy.iloc[:, hits_idx[3]] = data_copy.iloc[:, hits_idx[3]] / \
                np.log(max_charge)
        else:
            data_copy.iloc[:, hits_idx[3]] = data_copy.iloc[:, hits_idx[3]] / max_charge
            
    return data_copy

def set_window(df: pd.DataFrame, win_shape=(64, 64, 128), projection="3d",
               cube_pool="mean", projection_pool="max", range_y_sumed=None,
               range_hits_charge_sumed=None):
    """
    Función que dado un evento (df) se queda solo con una ventana de todo 
    el evento fija en el incio del evento, al ser la parte más importante del 
    mismo. Hace un cambio de sistema de referencia en los eje para que las
    posciones de los hits en la venta estén dados como índices entre 0 y 
    las dimensiones dadas por win_shape.

    También realiza las distintas proyecciones posibles con la que podemos 
    tratar las imágenes.

    Parameters
    ----------
    df: DataFrame del evento a tratar.
    win_shape: dimensiones de la ventana que se quiere situar.
    projection: tipo de proyección que se quiere realizar. Acepta "3d" si 
    queremos obtener imagenes en 3d. "z" si queremos proyectar el eje Z, 
    "x" si queremos proyectar el eje X e "y" si queremos proyectar el eje Y.
    cube_pool: tratamiento de distintos hits con misma posición espacial
    tras la definición del cubo.
    projection_pool: tratamiento de distintos hits con misma posición espacial
    tras la proyección del cubo.
    Return
    ----------
    df_win: DataFrame con la posición (como índices de los píxeles de la imagen
    final) y carga normalizada de los eventos dentro de la venta. 
    """
    names = ["hitsX", "hitsY", "hitsZ"]
    hits_idx = [df.columns.get_loc(i) for i in names]
    # eje x
    # vemos la dirección de drift:
    # obtenemos la diferencia de la posición en X de todos los hits con el 
    # inicial
    diff_x = df["hitsX"] - df["hitsX"].iloc[0]
    n_hits = diff_x.shape[0]
    # porcentaje de hits con X mayor que el incial
    positve_ratio = (diff_x > 0).sum() / n_hits
    # consideramos que la dirección es positiva si el ratio es mayor que 0.5
    if positve_ratio > 0.5:
        x_dir = "positive"
    else:
        x_dir = "negative"
    # pool incial (tras haber hecho el cubo)
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
        if x_dir == "positive":  # el hit incial se encuentra a la izquierda
            df_win.iloc[:, hits_idx[0]] = new_x + 10
        elif x_dir == "negative":  # el hit incial se encuentra a la derecha
            df_win.iloc[:, hits_idx[0]] = new_x + win_shape[0] - 10
    else:  # Si tenemos menos de 30 hits nos situamos en un punto medio
        df_win.iloc[:, hits_idx[0]] = new_x + int(win_shape[0] / 2)
    if not projection == "color":
        df_win.iloc[:, hits_idx[1]] = df_win["hitsY"] - df["hitsY"].iloc[1] + \
            int(win_shape[1] / 2)
    # Invertimos el eje y
    # df_win.iloc[:, hits_idx[1]] = win_shape[1] - df_win.iloc[:, hits_idx[1]] - 1

    df_win.iloc[:, hits_idx[2]] = df_win["hitsZ"] - df["hitsZ"].iloc[0] + 5
    # Invertimos el eje z
    # df_win.iloc[:, hits_idx[2]] = win_shape[2] - df_win.iloc[:, hits_idx[2]] - 1

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
        # invertimos los ejes "y" y "z" 
        df_win.iloc[:, hits_idx[1]] = win_shape[1] - \
            df_win.iloc[:, hits_idx[1]] - 1
        df_win.iloc[:, hits_idx[2]] = win_shape[2] - \
            df_win.iloc[:, hits_idx[2]] - 1
        image = np.zeros(win_shape)
        for index, row in df_win.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsY"])
            z = int(row["hitsZ"])
            image[x, y, z] = row["hitsCharge"]

        image = image.reshape((1, win_shape[2], win_shape[1], win_shape[0]))
        return df_win, image
    
    elif projection == "z":
        df_win = df_win.drop(["hitsZ"], axis=1)
        names = ["hitsX", "hitsY"]
        hits_idx = [df_win.columns.get_loc(i) for i in names]
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
        df_win.iloc[:, hits_idx[1]] = win_shape[1] - \
            df_win.iloc[:, hits_idx[1]] - 1
        image = np.zeros((win_shape[0], win_shape[1]))
        for index, row in df_win.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsY"])
            image[y, x] = row["hitsCharge"]
        image = image.reshape((1, win_shape[1], win_shape[0]))
        return df_win, image
    
    elif projection == "y":
        df_win = df_win.drop(["hitsY"], axis=1)
        names = ["hitsX", "hitsZ"]
        hits_idx = [df_win.columns.get_loc(i) for i in names]
        if projection_pool == "max":
            df_win = df_win.groupby(["hitsX", "hitsZ"]).max().reset_index()
        elif projection_pool == "mean":
            df_win = df_win.groupby(["hitsX", "hitsZ"]).mean().reset_index()
        
        df_win = df_win[
            np.logical_and(
                df_win["hitsX"] < win_shape[0],
                df_win["hitsX"] >= 0
            )
        ]
        df_win = df_win[
            np.logical_and(
                df_win["hitsZ"] < win_shape[2],
                df_win["hitsZ"] >= 0
            )
        ]
        # invertimos el z
        df_win.iloc[:, hits_idx[1]] = win_shape[2] - \
            df_win.iloc[:, hits_idx[1]] - 1
        image = np.zeros((win_shape[2], win_shape[0]))
        for index, row in df_win.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsZ"])
            image[y, x] = row["hitsCharge"]
        image = image.reshape((1, win_shape[2], win_shape[0]))
        return df_win, image
    
    elif projection == "x":
        df_win = df_win.drop(["hitsX"], axis=1)
        names = ["hitsY", "hitsZ"]
        hits_idx = [df_win.columns.get_loc(i) for i in names]
        if projection_pool == "max":
            df_win = df_win.groupby(["hitsY", "hitsZ"]).max().reset_index()
        elif projection_pool == "mean":
            df_win = df_win.groupby(["hitsY", "hitsZ"]).mean().reset_index()
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
        df_win.iloc[:, hits_idx[1]] = win_shape[2] - \
            df_win.iloc[:, hits_idx[1]] - 1
        image = np.zeros((win_shape[2], win_shape[1]))
        for index, row in df_win.iterrows():
            x = int(row["hitsY"])
            y = int(row["hitsZ"])
            image[y, x] = row["hitsCharge"]
        image = image.reshape((1, win_shape[2], win_shape[1]))
        return df_win, image
    
    elif projection == "color":
        # agrupamos por x y z, en los grupos tenemos puntos que coinciden en 
        # dicho plano y los sumamos
        y_min = df_win['hitsY'].min()
        df_win['hitsY_mod'] = df_win['hitsY'] - y_min

        df_win = df_win.groupby(["hitsX", "hitsZ"], as_index = False) \
            .sum() \
            .reset_index()
        # hacemos un cambio de rango para pasar a hue y saturación
        old_y = df_win.hitsY_mod
        old_charge = np.log(df_win.hitsCharge)
        y_min_all = range_y_sumed[0]
        y_min_all = y_min_all
        y_max_all = range_y_sumed[1]
        charge_min_all = range_hits_charge_sumed[0]
        charge_max_all = range_hits_charge_sumed[1]
        new_y = (old_y - y_min_all) / (y_max_all - charge_min_all) * 180
        new_charge = (old_charge - charge_min_all) * 255 / \
            (charge_max_all - charge_min_all)

        df_win.hitsY = new_y
        df_win.hitsCharge = new_charge

        names = ["hitsX", "hitsZ"]
        hits_idx = [df_win.columns.get_loc(i) for i in names]

        df_win = df_win[
            np.logical_and(
                df_win["hitsX"] < win_shape[0],
                df_win["hitsX"] >= 0
            )
        ]
        df_win = df_win[
            np.logical_and(
                df_win["hitsZ"] < win_shape[2],
                df_win["hitsZ"] >= 0
            )
        ]
        df_win.iloc[:, hits_idx[1]] = win_shape[2] - \
            df_win.iloc[:, hits_idx[1]] - 1
        image = np.zeros((win_shape[2], win_shape[0], 3))
        for index, row in df_win.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsZ"])
            saturation = 255 / 2
            hue = row["hitsY"]
            luminosity = row["hitsCharge"]
            image[y, x, 0] = hue
            image[y, x, 1] = luminosity
            image[y, x, 2] = saturation
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
        image = image.transpose((2, 0, 1))
        return df_win, image
    
    else:
        raise Exception("Projection name not valid")

class Data():
    """
    Clase que crea en disco imágenes para cada evento a partir del conjnto de
    datos dado, tabla con todos los eventos, estas imagenes son tras aplicar 
    la función rescale_axis, por lo que se mantien todos los eventos y lo 
    único que se ha procesado es la resolución de las imágenes.
    """
    def __init__(
        self, seed_=123, cube_shape_x=1000, win_shape=(62, 62, 128),
        projection="y", cube_pool="mean", projection_pool="max", log_trans=False
    ) -> None:
        
        # Definimos la forma de los ejes apartir del eje x
        cube_shape = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
        # Definimos atributos de entrada
        self.cube_shape = cube_shape
        self.win_shape = win_shape
        self.projection = projection
        self.cube_pool = cube_pool
        self.projection_pool = projection_pool
        self.seed = seed_
        self.log_trans = log_trans
        # Nombre del directorio, depende de los parámetros
        self.dir_path = IMG_PATH + f"cube_{cube_shape_x}"
        self.dir_path_test = os.path.join(self.dir_path, "test")
        
        self.is_new_data = not os.path.isdir(self.dir_path)

        if self.is_new_data:
            # Si no existe el directorio creamos las imagenes
            os.mkdir(self.dir_path)
            os.mkdir(self.dir_path_test)
            all_events = self.dump_ids()
            self.create_cube(all_events)
        else:
            # Si existe, solo cargamos las etiquetas
            with open(self.dir_path + "/ids.pickle", "rb") as file:
                self.ids_all_train, self.ids_test  = pickle.load(file)
        self.ids_train, self.ids_val = train_test_split(
            self.ids_all_train, random_state=self.seed, test_size=0.1
        )
            

    def dump_ids(self):
        """
        Método para cargar el dataframe con todos los eventos, con dos opciones
        valición o prueba final donde se incluyen los datos de valición. 
        """
        print("loading data:")
        all_events_train = pd.concat(
            [pd.read_csv("data/photons_train.csv"),
            pd.read_csv("data/electrons_train.csv")],
            axis=0
        )
        all_events_test = pd.concat(
            [pd.read_csv("data/photons_test.csv"),
            pd.read_csv("data/electrons_test.csv")],
            axis=0
        )
        all_events = pd.concat(
            [all_events_train, all_events_test],
                axis=0
        )
        self.ids_all_train = all_events_train[["Event", "PDGcode"]]\
            .drop_duplicates()
        self.ids_test = all_events_test[["Event", "PDGcode"]].drop_duplicates()
        # guardamos en un fichero los ids de los eventos separados en train y
        # test
        with open(self.dir_path + "/ids.pickle", "wb") as file:
            pickle.dump((self.ids_all_train, self.ids_test), file)
        return all_events

    def create_cube(self, all_events):
        self.mins_maxs, self.max_charge = get_min_max(
            all_events,
            self.ids_all_train
        )
        all_events = rescale_axis(
            all_events,
            self.mins_maxs,
            self.max_charge,
            self.cube_shape,
            self.log_trans
        )
        print("creating train images:")
        self.dump_img(all_events, self.ids_all_train, self.dir_path)
        print("creating test images:")
        self.dump_img(all_events, self.ids_test, self.dir_path_test)

    def dump_img(self, all_events, ids, path):
        """
        Itear por todas los eventos y los carga a disco
        """
        i = 0
        for index, row in ids.iterrows():
            df = all_events.query(
                f"Event == {row['Event']} and PDGcode == {row['PDGcode']}"
            ).copy() 
            df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
            if row['PDGcode'] == 11:
                label = 0
            elif row['PDGcode'] == 22:
                label = 1
            file_name = os.path.join(
                path, f"{row['Event']}_{row['PDGcode']}.pickle"
            ) 
            with open(file_name, "wb") as file:
                pickle.dump((df, label), file)
            if i % 2000 == 0:
                print(f"created {i} images")
            i += 1
        
class Cascadas(Dataset):

    def __init__(
        self, seed_=123, train: bool=True, validation: bool=True,
        transform=None, cube_shape_x=1000, win_shape=(62, 62, 128),
        projection="y", cube_pool="mean", projection_pool="max", log_trans=False
    ) -> None:
        """
        Clase que...
        """
        super().__init__()
        # initialization atributes
        self.win_shape = win_shape
        self.seed = seed_
        # other dimensions shapes
        cube_shape = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
        self.cube_shape = cube_shape
        self.projection = projection
        self.cube_pool = cube_pool
        self.train = train
        self.projection_pool = projection_pool
        self.transform = transform
        self.log_trans = log_trans
        # validation stage
        data = Data(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection=self.projection,
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool,
            log_trans=self.log_trans
        )
        self.dir_path = data.dir_path
        if validation:
            if train:
                self.ids = data.ids_train
            else:
                self.ids = data.ids_val
        else:
            if train:
                self.ids = data.ids_all_train
            else:
                self.dir_path = data.dir_path_test
                self.ids = data.ids_test
        
    def __len__(self):
        """
        Longitud del objeto como el nº de imagenes en el conjunto de datos.
        """
        return self.ids.shape[0]
    
    def __getitem__(self, idx):
        """
        Se devuelve una tupla con la imagen y su etiqueta
        """
        event = self.ids.iloc[idx, 0]
        label = self.ids.iloc[idx, 1]
        file_name = os.path.join(self.dir_path, f"{event}_{label}.pickle")
        # read the file and cut the wholo imagen
        with open(file_name, "rb") as file:
            df, label = pickle.load(file)
        _, img = set_window(
            df,
            projection=self.projection,
            win_shape=self.win_shape,
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool
        )
        # apply transformation
        # if self.transform:
        #     img = self.transform(img)
        return img, label

    def plot_simple(self, idx):
        """
        plot of the image given by the idx
        """
        img, _ = self[idx]
        print(self.ids.iloc[idx, 0], self.ids.iloc[idx, 1])
        if not self.projection == "3d":
            plt.figure()
            plt.imshow(img[0, :, :], cmap="gray")
            plt.show()


    def plot(self, idx):
        """
        Some plot of the even given by the idx. 

        1. Raw plot
        2. Plot after changing the units in the spatial axis
        3. Plot after setting the window
        4. Plot of the final matrix (image given to cnn's) if not 3d imagen
        """
        # We need to load all the events
        if not hasattr(self, 'all_events'):
                self.all_events = pd.concat(
                    [pd.read_csv("data/photons.csv"),
                    pd.read_csv("data/electrons.csv")],
                    axis=0
                )
        # get event id, label and file_name
        event = self.ids.iloc[idx, 0]
        label = self.ids.iloc[idx, 1]
        file_name = os.path.join(self.dir_path, f"{event}_{label}.pickle")
        # read file and set window
        with open(file_name, "rb") as file:
            df, _ = pickle.load(file)
            df_2, _ = set_window(
                df,
                projection="3d",
                win_shape=self.win_shape,
                cube_pool=self.cube_pool,
                projection_pool=self.projection_pool
            )
        # 1º plot
        vistas_detector(self.all_events, event, label, False)
        # 2º plot
        plot_vistas(df, None)
        # 3º plot
        plot_vistas(df_2, self.win_shape)
        img, _ = self[idx]
        print(event, label)
        # 4º plot
        if not self.projection == "3d":
            plt.figure()
            plt.imshow(img[0, :, :], cmap="gray")
            plt.show()

class DataFast():
    """
    Clase que crea en disco imágenes para cada evento a partir del conjnto de
    datos dado, tabla con todos los eventos, estas imagenes son tras quedarnos 
    únicamente con la ventana de observación
    """
    def __init__(
        self, seed_=123, cube_shape_x=1000, win_shape=(62, 62, 128),
        projection="y", cube_pool="mean", projection_pool="max", log_trans=False
    ) -> None:
        
        # Definimos la forma de los ejes apartir del eje x
        cube_shape = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
        # Definimos atributos de entrada
        self.cube_shape = cube_shape
        self.win_shape = win_shape
        self.projection = projection
        self.cube_pool = cube_pool
        self.projection_pool = projection_pool
        self.seed = seed_
        self.log_trans = log_trans
        # Nombre del directorio, depende de los parámetros
        self.dir_path = IMG_PATH + f"{cube_shape_x}_{projection}" +\
                f"_{cube_pool}_{projection_pool}_{win_shape[0]}" +\
                f"_{win_shape[1]}_{win_shape[2]}_{log_trans}"
        self.dir_path_test = os.path.join(self.dir_path, "test")
        
        self.is_new_data = not os.path.isdir(self.dir_path)
        if self.is_new_data:
            # Si existe el directorio creamos las imagenes
            os.mkdir(self.dir_path)
            os.mkdir(self.dir_path_test)
            all_events = self.dump_ids()
            self.create_img(all_events)
        else:
            # Si existe, solo cargamos las etiquetas
            with open(self.dir_path + "/ids.pickle", "rb") as file:
                self.ids_all_train, self.ids_test  = pickle.load(file)
        self.ids_train, self.ids_val = train_test_split(
            self.ids_all_train, random_state=self.seed, test_size=0.1
        )

    def dump_ids(self):
        """
        Método para cargar el dataframe con todos los eventos tanto train como
        test. Tambien se guardan en memoria los ids de train y test por separado
        """
        print("loading data:")
        all_events_train = pd.concat(
            [pd.read_csv("data/photons_train.csv"),
            pd.read_csv("data/electrons_train.csv")],
            axis=0
        )
        all_events_test = pd.concat(
            [pd.read_csv("data/photons_test.csv"),
            pd.read_csv("data/electrons_test.csv")],
            axis=0
        )
        # juntamos los datos
        all_events = pd.concat(
            [all_events_train, all_events_test],
                axis=0
        )
        # obtenemos los ids que son un dataframe con el PDGCode y Event
        self.ids_all_train = all_events_train[["Event", "PDGcode", "Energy"]]\
            .drop_duplicates()
        self.ids_test = all_events_test[["Event", "PDGcode", "Energy"]]\
            .drop_duplicates()
        # guardamos en un fichero los ids de los eventos separados en train y
        # test
        with open(self.dir_path + "/ids.pickle", "wb") as file:
            pickle.dump((self.ids_all_train, self.ids_test), file)
        return all_events

    def create_img(self, all_events):
        """
        Método que dado el dataframe con los todos los eventos, crea las imágenes
        de todos los eventos y las carga en disco
        """
        # obtenemos usando solo train los mínimos y máximos de la posición y 
        # máximo de la carga
        self.mins_maxs, self.max_charge, self.range_y, self.range_charge = get_min_max(
            all_events,
            self.ids_all_train
        )
        # Cambiamos el valor de los ejes para que sean números enteros sin 
        # perder información
        # y de la carga para que la máxima (* 1.15) sea igual a 1 y este acotada
        # por este valor 
        all_events = rescale_axis(
            all_events,
            self.mins_maxs,
            self.max_charge,
            self.cube_shape,
            self.projection,
            self.log_trans
        )
        # cargamos en disco
        print("creating training images:")
        self.dump_img(all_events, self.ids_all_train, self.dir_path, 
                      self.range_y, self.range_charge)
        print("creating test images:")
        self.dump_img(all_events, self.ids_test, self.dir_path_test,
                      self.range_y, self.range_charge)

    def dump_img(self, all_events, ids, dir_path, range_y, range_charge):
        """
        Itera por todas los eventos y los carga a disco
        """
        i = 0
        # iteramos por las filas del dataframe con las ids
        for index, row in ids.iterrows():
            # nos quedamos con el evento
            df = all_events.query(
                f"Event == {row['Event']} and PDGcode == {row['PDGcode']}"
            ).copy() 
            df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
            _, img = set_window(
                df,
                projection=self.projection,
                win_shape=self.win_shape,
                cube_pool=self.cube_pool,
                projection_pool=self.projection_pool,
                range_hits_charge_sumed=range_charge,
                range_y_sumed=range_y
            )
            # cambiamos los códigos de manera que el electrón sea 0 y el foton 1
            if row['PDGcode'] == 11:
                label = 0
            elif row['PDGcode'] == 22:
                label = 1
            # cargamos con pickle
            file_name = os.path.join(
                dir_path, f"{int(row['Event'])}_{int(row['PDGcode'])}.pickle"
            ) 
            with open(file_name, "wb") as file:
                pickle.dump((img, label), file)
            if i % 2000 == 0:
                print(f"created {i} images")
            i += 1

class CascadasFast(Dataset):
    """
    Clase con los datos de nuestro problema que hereda de la clase Dataset de 
    torch para que funcione con los dataloader.
    """
    def __init__(
        self, seed_=123, train: bool=True, validation: bool=True,
        transform=None, cube_shape_x: int=1000, win_shape=(62, 62, 128),
        projection="y", cube_pool="mean", projection_pool="max", log_trans=False
    ) -> None:
        """
        Clase que...
        """
        super().__init__()
        self.win_shape = win_shape
        self.seed = seed_
        # obtenemos del valor de cube_shape_x el valor en los otros ejes para
        # que todos tengan las mismas dimensiones. Eje x e y 400 eje z 500.
        cube_shape = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
        self.cube_shape = cube_shape
        self.projection = projection
        self.cube_pool = cube_pool
        self.train = train
        self.projection_pool = projection_pool
        self.transform = transform
        self.log_trans = log_trans
        # usamos la clase Data que nos proporciona el diretorio de las imágenes
        # los ids y en caso de necesitarlo descarga las imágenes
        data = DataFast(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection=self.projection,
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool,
            log_trans=self.log_trans

        )
        self.dir_path = data.dir_path
        # obtenemos los ids que toquen, pueden ser los de validacion 
        # (train validacion)
        # o todos los de train y test
        # (all_train test) 
        if validation:
            if train:
                self.ids = data.ids_train
            else:
                self.ids = data.ids_val
        else:
            if train:
                self.ids = data.ids_all_train
            else:
                self.dir_path = data.dir_path_test
                self.ids = data.ids_test
        
    def __len__(self):
        """
        Longitud del objeto como el nº de imagenes en el conjunto de datos.
        """
        return self.ids.shape[0]
    
    def __getitem__(self, idx):
        """
        Se devuelve una tupla con la imagen y su etiqueta
        """
        event = self.ids.iloc[idx, 0]
        label = self.ids.iloc[idx, 1]
        energy = self.ids.iloc[idx, 2]
        # cargamos la imagen de disco
        file_name = os.path.join(self.dir_path, f"{event}_{label}.pickle")
        with open(file_name, "rb") as file:
            img, label = pickle.load(file)
        img = torch.tensor(img)
        # apply transformation
        if self.transform:
            img = self.transform(img)
        return img, label, energy

    def plot_simple(self, idx):
        """
        plot of the image given by the idx
        """
        img, _, _ = self[idx]
        print(self.ids.iloc[idx, 0], self.ids.iloc[idx, 1], self.ids.iloc[idx, 2])
        if self.projection == "color":
            img = img.numpy()
            img = img.transpose((1, 2, 0))
            plt.figure()
            plt.imshow(img)
            plt.show()
        elif not self.projection == "3d":
            plt.figure()
            plt.imshow(img[0, :, :], cmap="gray")
            plt.show()
        else:
            fig = plt.figure(figsize=(16,11))
            ax = fig.add_subplot(111, projection='3d')
            cmap = ListedColormap(sns.color_palette("flare").as_hex())
            z, x, y = img.nonzero()
            sc = ax.scatter(
                xs=x,
                ys=y,
                zs=z,
                alpha=0.8,
                c=z,
                cmap=cmap
    )

class CascadasMulti(Dataset):
    """
    Clase con los datos de nuestro problema que hereda de la clase Dataset de 
    torch para que funcione con los dataloader.
    """
    def __init__(
        self, seed_=123, train: bool=True, validation: bool=True,
        transform=None, cube_shape_x: int=1000, win_shape=(62, 62, 128),
        cube_pool="mean", projection_pool="max", log_trans=False
    ) -> None:
        """
        Clase que...
        """
        super().__init__()
        self.win_shape = win_shape
        self.seed = seed_
        # obtenemos del valor de cube_shape_x el valor en los otros ejes para
        # que todos tengan las mismas dimensiones. Eje x e y 400 eje z 500.
        cube_shape = (cube_shape_x, cube_shape_x, cube_shape_x * 5 / 4)
        self.cube_shape = cube_shape
        self.cube_pool = cube_pool
        self.train = train
        self.projection_pool = projection_pool
        self.transform = transform
        self.log_trans = log_trans
        # usamos la clase Data que nos proporciona el diretorio de las imágenes
        # los ids y en caso de necesitarlo descarga las imágenes
        data_x = DataFast(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection='x',
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool,
            log_trans=self.log_trans

        )
        self.dir_path_x = data_x.dir_path
        data_y = DataFast(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection='y',
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool,
            log_trans=self.log_trans

        )
        self.dir_path_y = data_y.dir_path
        data_z = DataFast(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection='z',
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool,
            log_trans=self.log_trans

        )
        self.dir_path_z = data_z.dir_path
        # obtenemos los ids que toquen, pueden ser los de validacion 
        # (train validacion)
        # o todos los de train y test
        # (all_train test) 
        if validation:
            if train:
                self.ids_x = data_x.ids_train
                self.ids_y = data_y.ids_train
                self.ids_z = data_z.ids_train
            else:
                self.ids_x = data_x.ids_val
                self.ids_y = data_y.ids_val
                self.ids_z = data_z.ids_val
        else:
            if train:
                self.ids_x = data_x.ids_all_train
                self.ids_y = data_y.ids_all_train
                self.ids_z = data_z.ids_all_train
            else:
                self.dir_path_x = data_x.dir_path_test
                self.dir_path_y = data_y.dir_path_test
                self.dir_path_z = data_z.dir_path_test
                self.ids_x = data_x.ids_test
                self.ids_y = data_y.ids_test
                self.ids_z = data_z.ids_test
        
    def __len__(self):
        """
        Longitud del objeto como el nº de imagenes en el conjunto de datos.
        """
        return self.ids_x.shape[0]
    
    def __getitem__(self, idx):
        """
        Se devuelve una tupla con la imagen y su etiqueta
        """
        # TODO comprobar que para las tres vistas este saliendo lo mismo
        event = self.ids_y.iloc[idx, 0]
        label = self.ids_y.iloc[idx, 1]
        energy = self.ids_y.iloc[idx, 2]
        # cargamos la imagen de disco
        img_x, _ = self.get_img(event, label, 'x')       
        img_y, new_label = self.get_img(event, label, 'y')       
        img_z, _ = self.get_img(event, label, 'z')       
        # apply transformation
        if self.transform:
            img_x = self.transform(img_x)
            img_y = self.transform(img_y)
            img_z = self.transform(img_z)
        return (img_x, img_y, img_z), new_label, energy

    def get_img(self, event, label, projection):
        if projection == 'x':
            file_name = os.path.join(self.dir_path_x, f"{event}_{label}.pickle")
        elif projection == 'y':
            file_name = os.path.join(self.dir_path_y, f"{event}_{label}.pickle")
        elif projection == 'z':
            file_name = os.path.join(self.dir_path_z, f"{event}_{label}.pickle")

        with open(file_name, "rb") as file:
            img, new_label = pickle.load(file)
        img = torch.tensor(img)
        return img, new_label

    def plot_simple(self, idx):
        """
        plot of the image given by the idx
        """
        img, _, _ = self[idx]
        print(self.ids.iloc[idx, 0], self.ids.iloc[idx, 1], self.ids.iloc[idx, 2])
        if self.projection == "color":
            img = img.numpy()
            img = img.transpose((1, 2, 0))
            plt.figure()
            plt.imshow(img)
            plt.show()
        elif not self.projection == "3d":
            plt.figure()
            plt.imshow(img[0, :, :], cmap="gray")
            plt.show()
        else:
            fig = plt.figure(figsize=(16,11))
            ax = fig.add_subplot(111, projection='3d')
            cmap = ListedColormap(sns.color_palette("flare").as_hex())
            z, x, y = img.nonzero()
            sc = ax.scatter(
                xs=x,
                ys=y,
                zs=z,
                alpha=0.8,
                c=z,
                cmap=cmap
    )
# %%
if __name__ == "__main__":
    # cascada = CascadasFast(cube_shape_x=1000)
    # img, label = cascada[7]
    # cascada.plot_simple(7)
    import torchvision.transforms as T
    transform = T.Compose([
        T.Lambda(lambda x: x.type(torch.float)),
        T.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    cascada = CascadasFast(cube_shape_x=1000, projection="color", 
                           win_shape=(62, 62, 128), transform=transform)
    # cascada.plot_simple(469)
    # cascada_2 = CascadasFast(cube_shape_x=1000, projection="color", 
    #                        win_shape=(62, 62, 128))
    # cascada_2.plot_simple(5)
    # %%
    img, _, _ = cascada[469]
    img.max()
# %%
