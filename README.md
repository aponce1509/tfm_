# Separación de cascadas electromagnéticas en el experimento SBND mediante técnicas de Machine Learning

* Código desarrollado para el trabajo de Fin de Master del master de ciencia de datos e ingeniería de computadores por la universidad de Granada.

* En el repositorio no se han añadido los datos, en caso de querer usarlos mandar un correo electrónico a: aponce1509@hotmail.com.

* En la carpeta scripts se tiene todo el código desarrollado, donde se incluyen distintas carpetas con los distintos experimentos realizados.

* Se puede llamar al script main.py para ejecutar el entrenamiento de un modelo. Se tienen distintos modelos y un ejemplo de ejecución sería:

```
python scripts/main.py EffNet simple
```

* En el archivo path.py se tienen que definir una serie de variables globales:

    1. `IMG_PATH`: ruta donde se tienen las imágenes.
    2. `MODEL_PATH`: ruta para MLflow.
    3. `TORCH_PATH`: ruta donde se guardan los modelos de pytorch.

## Código

Tenemos distintos scripts donde cada uno se encarga de contener una parte del flujo de trabajo realizado:

1. `main.py`: programa que ejecuta el flujo completo de entrenamiento, simple y de optuna. 
1. `path.py`: Se definen las variables globales.
1. `train_test_split.py`: divide el conjunto de datos en un conjunto de entrenamiento y en otro de test.
1. `graficas_pdf.py`: creación de gráficas usando los datos en bruto para los distintos eventos
2. `eda.py` y `eda_2.py` tienen código desarrollado para la realización de imágenes y para el estudio de los datos.
1. `utils_win_cube_copy.py`: se realiza todo el cauce de creación de imágenes. Se incluyen distintas representaciones de los datos.
1. `model_run.py`: se realiza el flujo de entrenamiento completo.
1. `_models.py`: se definen todos los modelos de DL usados.
1. `utils_test.py`: funciones para realizar una batería de experimentos.
1. `ensemble.py`: funciones para realizar el ensemble
1. `utils_study_res.py`, `utils_res_ensemble.py`, `res_study.py`: funciones y programas para realizar el estudio de los modelos.
1. `new_data.py`: genera un nuevo conjunto de datos usando un modelo entrenado para cada proyección.
1. `grad_cam.py`, `features.py`, `misc_functions.py`: WIP. Son programas para implementar grad cam que no se han usado en el trabajo y están en desarrollo.

Por otro lado, se tienen distintas carpetas donde se realizan los experimentos. Las que tienen el prefijo `exp` recogen programas que realizan el entrenamiento en una representación concreta o usando un modelo concreto. En su interior se tiene un script: `params.py` donde se definen los parámetros para realizar el entrenamiento.

El resto de carpetas contienen scripts para la realización de los experimentos recogidos en la memoria del TFM. 
