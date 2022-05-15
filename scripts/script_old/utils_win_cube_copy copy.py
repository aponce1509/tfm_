# %%
from matplotlib import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
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

def get_min_max(data, ids_train=None):
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
        data_train = data    
    # obtenemos los máximos y mínimos
    min_max = []
    min_max.append(data_train.hitsX.min() - 5)
    min_max.append(data_train.hitsX.max() + 5)
    min_max.append(data_train.hitsY.min() - 5)
    min_max.append(data_train.hitsY.max() + 5)
    min_max.append(data_train.hitsZ.min() - 5)
    min_max.append(data_train.hitsZ.max() + 5)
    max_charge = data_train.hitsCharge.max() * 1.15
    return min_max, max_charge

def rescale_axis(data, mins_maxs, max_charge, cube_shape=(2000, 2000, 2500)):
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
    names = ["hitsX", "hitsY", "hitsZ", "hitsCharge"]
    # Obtenemos sus ids
    hits_idx = [data.columns.get_loc(i) for i in names]
    # Cambio del eje X
    data.iloc[:, hits_idx[0]] = round(((data["hitsX"] - mins_maxs[0]) / 
                     (mins_maxs[1] - mins_maxs[0])) * cube_shape[0], 0).\
                         astype(int)
    # Cambio del eje Y
    data.iloc[:, hits_idx[1]] = round(((data["hitsY"] - mins_maxs[2]) / 
                     (mins_maxs[3] - mins_maxs[2])) * cube_shape[1], 0).\
                         astype(int)
    # Cambio del eje Z
    data.iloc[:, hits_idx[2]] = round(((data["hitsZ"] - mins_maxs[4]) / 
                     (mins_maxs[5] - mins_maxs[4])) * cube_shape[2], 0).\
                         astype(int)
    # Cambio de la charga
    data.iloc[:, hits_idx[3]] = data.iloc[:, hits_idx[3]] / max_charge
    # En caso que haya valores mayores de la carga máxima (puede darse en 
    # validación) se fija al valor máximo de forma que la carga este acotada
    data.iloc[data["hitsCharge"] >= max_charge, hits_idx[3]] = max_charge 
    return data

def set_window(
        df, win_shape=(64, 64, 128), projection="3d", cube_pool="mean",
        projection_pool="max"
    ):
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

    df_win.iloc[:, hits_idx[1]] = df_win["hitsY"] - df["hitsY"].iloc[1] +\
        int(win_shape[1] / 2)
    # Invertimos el eje y
    df_win.iloc[:, hits_idx[1]] = win_shape[1] - df_win.iloc[:, hits_idx[1]] - 1

    df_win.iloc[:, hits_idx[2]] = df_win["hitsZ"] - df_win["hitsZ"].min()
    # Invertimos el eje z
    df_win.iloc[:, hits_idx[2]] = win_shape[2] - df_win.iloc[:, hits_idx[2]] - 1

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
        image = np.zeros(win_shape)
        for index, row in df_win.iterrows():
            x = int(row["hitsX"])
            y = int(row["hitsY"])
            z = int(row["hitsZ"])
            image[x, y, z] = row["hitsCharge"]
            return df_win, image
    elif projection == "z":
        df_win = df_win.drop(["hitsZ"], axis=1)
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
        image = np.zeros((win_shape[0], win_shape[1]))
        for index, row in df_win.iterrows():
            y = int(row["hitsX"])
            x = int(row["hitsY"])
            image[x, y] = row["hitsCharge"]
        image = image.reshape((1, win_shape[0], win_shape[1]))
        return df_win, image
    elif projection == "y":
        df_win = df_win.drop(["hitsY"], axis=1)
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
        image = np.zeros((win_shape[2], win_shape[0]))
        for index, row in df_win.iterrows():
            y = int(row["hitsX"])
            x = int(row["hitsZ"])
            image[x, y] = row["hitsCharge"]
        image = image.reshape((1, win_shape[2], win_shape[0]))
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
        projection="y", cube_pool="mean", projection_pool="max"
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
            self.cube_shape
        )
        print("creating train images:")
        self.dump_img(all_events, self.ids_all_train, self.dir_path)
        print("creating test images:")
        self.dump_img(all_events, self.ids_test, self.dir_path_test)

    def dump_img(self, all_events, ids, path):
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
        projection="y", cube_pool="mean", projection_pool="max"
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
        # validation stage
        data = Data(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection=self.projection,
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool
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
    datos dado, tabla con todos los eventos, estas imagenes son tras aplicar 
    la función rescale_axis, por lo que se mantien todos los eventos y lo 
    único que se ha procesado es la resolución de las imágenes.
    """
    def __init__(
        self, seed_=123, cube_shape_x=1000, win_shape=(62, 62, 128),
        projection="y", cube_pool="mean", projection_pool="max"
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
        # Nombre del directorio, depende de los parámetros
        self.dir_path = IMG_PATH + f"{cube_shape_x}_{projection}" +\
                f"_{cube_pool}_{projection_pool}_{win_shape[0]}" +\
                f"_{win_shape[1]}_{win_shape[2]}"
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

    def create_img(self, all_events):
        self.mins_maxs, self.max_charge = get_min_max(
            all_events,
            self.ids_all_train
        )
        all_events = rescale_axis(
            all_events,
            self.mins_maxs,
            self.max_charge,
            self.cube_shape
        )
        
        print("creating training images:")
        self.dump_img(all_events, self.ids_all_train, self.dir_path)
        print("creating test images:")
        self.dump_img(all_events, self.ids_test, self.dir_path_test)

    def dump_img(self, all_events, ids, dir_path):
        i = 0
        for index, row in ids.iterrows():
            df = all_events.query(
                f"Event == {row['Event']} and PDGcode == {row['PDGcode']}"
            ).copy() 
            df = df.drop(["Event", "PDGcode", "Energy"], axis=1)
            try:

                _, img = set_window(
                    df,
                    projection=self.projection,
                    win_shape=self.win_shape,
                    cube_pool=self.cube_pool,
                    projection_pool=self.projection_pool
                )
            except:
                print(f"{row}")
            if row['PDGcode'] == 11:
                label = 0
            elif row['PDGcode'] == 22:
                label = 1
            file_name = os.path.join(
                dir_path, f"{row['Event']}_{row['PDGcode']}.pickle"
            ) 
            with open(file_name, "wb") as file:
                pickle.dump((img, label), file)
            if i % 2000 == 0:
                print(f"created {i} images")
            i += 1

class CascadasFast(Dataset):

    def __init__(
        self, seed_=123, train: bool=True, validation: bool=True,
        transform=None, cube_shape_x=1000, win_shape=(62, 62, 128),
        projection="y", cube_pool="mean", projection_pool="max"
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
        data = DataFast(
            seed_=self.seed,
            cube_shape_x=cube_shape_x,
            win_shape=self.win_shape,
            projection=self.projection,
            cube_pool=self.cube_pool,
            projection_pool=self.projection_pool
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
            img, label = pickle.load(file)
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

if __name__ == "__main__":
    # cascada = CascadasFast(cube_shape_x=1000)
    # img, label = cascada[7]
    # cascada.plot_simple(7)
    cascada = CascadasFast(cube_shape_x=1000, projection="3d")
    # cascada.plot_simple(7)
# #%%
# data = np.random.random(size=(3, 2))
# z, x = data.nonzero()
# print(z.shape, x.shape)
# # %%
# data
# x