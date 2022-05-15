# Versión 0.1
a
Incluye:

* Gráficas (vistas y 3d) de los eventos
* División del conjunto de entrenamiento y prueba en dos archivos
* Clase Cascadas definida como la clase DataSet de torch y permite cargar las imágenes de train y test así como hacer validación en train y test.
* Incluye dos formas de tratar las imágenes para el modelo:
    - Como una imagen 3d
    - Como una proyección en uno de los ejes
* Incluye también distintas formas de hacer pool
* Modelo esquemático de una CNN con optuna y MLflow
* Un archivo de EDA para ver los rango de valores que se toma tanto en energía como en posición

En la siguiente versión se quiere añadir:

* Mejorar los comentarios
* CNN 2d con una sola proyección completamente funcional y con mensajes de evolución durante el entrenamiento.
* Arreglar un fallo en el método plot de la clase Cascadas

## Ejecución

Para la ejecución ahora mismo se puede o ejecutar el archivo graficas_pdf.py para generar gráficas del evento. Lo único que hay que hacer es llamar a la función vistas_y_3d y darle en número del evento así como el tipo de partícula.

También se puede ejecutar el archivo eda.py para ver algunas gráficas que describen los datos. Por último se puede ejecutar el archivo utils.py que si se ejecuta te muestra los datos que estamos obteniendo tras hacer el tratamiento. La primera gráfica es los datos tras hacer el cambio de referencia y de unidades. La segunda es la proyección en el plano xy del evento ya como una matriz y lo que se representa es dicha matriz.

## graficas_pdf.py

Incluye funciones para representar los eventos tanto en 3d como con las vistas. Se definen 3 funciones una para las vistas otra para 3d y otra para las dos en conjunto. Se puede hacer que la función guarde además las gráficas como pdf.

## eda.py 

Cuaderno para ir realizando gráficos y observaciones sobre los datos

## test_train_split.py

Realiza la división del conjunto de datos inicial en uno con los eventos de entrenamiento y otros con los de test con la idea de solo usarlos al final para ver los resultados. Para la selección del modelo final ajustando parámetros lo que haremos es crear un conjunto de validación dentro del conjunto de entrenamiento

## utils.py

Se define la clase fundamental con la que vamos a trabajar. Se con la clase DataSet de torch como padre con la idea que trabaje bien con las funcionalidades propias de torch.

Inicialmente definimos 4 funciones necesarias para tratar los csv iniciales. Una que hace gráficos. Luego tenemos dos necesarias para hacer el cambio de unidades de los ejes. Otra función para colocar la venta que vamos a ver de los datos ya que no podemos mostrar todo a la red.

Los parámetros son:

* seed_: Semilla para reproducibilidad
* train: Booleano si estamos con en conjunto de entrenamiento o de prueba
* validation: Boolena si estamos haciendo validación o no. Si estamos en validación tenemos un conjunto de test obtenido del conjunto de entrenamiento inicial.
* transform: transformador de pytorch por si queremos normalizar o convertir en tensores. En cierta forma permite meter cosas de preprocesamiento.
* cube_shape: dimensiones del cubo donde vamos a colocar todos los eventos. Por así decirlo, este cubo es el sistema de referencia donde vamos a trabajar. Se define de manera que se hace una cambio de referencia y de unidades de los ejes iniciales de manera que los valores de los puntos sean números enteros. Para esto se están redondeando los valores tras hacer el cambio de unidades por lo que es necesario hacer una especie de pool para los puntos que coincidan.
* win_shape: dimensiones de la venta tridimensional que vamos a observar
* proyection: tipo de projection que queremos usar. Incluye 3d que realmente no hace ninguna proyección, y sobre el eje z
* cube_pool: tipo de pool que se hace cuando se define el cubo. Incluye media y maximo
* projection_pool: tipo de pool que se hace cuando se proyecta. Incluye media y maximo

## cnn_2d_torch.py

Modelo esquemático de la CNN con optuna y mlflow tengo que depurarla y comprobar que funciona correctamente.

## parameters.py

Recoge los distintos parámetros que tenemos en nuestro modelo y tratamiento de datos para que se pueda hacer un seguimiento simple con MLflow.

# Versión 0.2.1

Arrglo de fallos y cambios menores:

- utils.py
    * Fallo a la hora de elegir la dirección de avance de la cascada en el eje
    x. Ahora se resta el inicial con todos y en función del promedio de los signos 
    se toma un dirección u otra.
    * Se tiene cuidado con los eventos poco energéticos donde la venta se coloca 
    en medio
    * Las imágenes se devuelven con la forma (1, cx, cy)
    * Se proyecta por defecto en y
    * Se hace que el cubo tenga las mismas unidades en los tres ejes (hay que 
    tener cuidado con el eje z por que mide 500cm mientras que los otros miden
    400)
    * Por defecto ahora el win_shape es 62x62x128
    * Añadida normalización con respecto a la carga (usando el conjunto 
    de entrenamiento)
    * Quitada opción de usar tranforms de torch ya que daban error a la 
    de hacer la transformación a tensor ya que cambia el orden de las 
    dimensiones
    * las etiquetas de salida ahora son 0 (electrón) o 1 (fotón)
    * el get_item ahora solo devuelve la imagen y la etiqueta
    * El método plot da muchos fallos se queda únicamente con mostrar 
    la proyección (temporalmente)  
- eda.py
    * Añadido estudio de los eventos con pocos hits
- parameters.py
    * Añadido como parámetros n_epochs y dropout

## cnn_2d.py

Nuevo archivo con la una red neuronal cnn simple depurado (parte del archivo
anterior cnn_2d_torch.py). Incluye tensorboard para hacer un seguimiento 
del entrenamiento y validación de la red. Esta depurada para la run simple
sin ajuste de hiperparámetros ni mlflow.

Las funciones las definiré en una versión más depurada de este archivo