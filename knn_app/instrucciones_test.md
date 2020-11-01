# Trabajo Final - Grupo 5 - Inteligencia Artificial 2020 - K-Nearest Neighbors


## ¿Que método de validación elegir?
La **validación cruzada** es una técnica para evaluar modelos de Machine Learning mediante el entrenamiento de varios modelos de Machine Learning en subconjuntos de los datos de entrada disponibles y evaluarlos con el subconjunto complementario de los datos.

En la validación cruzada de K iteraciones se dividen los datos de entrada en K subconjuntos de datos (también conocido como iteraciones). Puede entrenar un modelo de ML en todos menos uno (k-1) de los subconjuntos y, a continuación, evaluar el modelo en el subconjunto que no se ha utilizado para el entrenamiento. Este proceso se repite K veces, con un subconjunto diferente reservado para la evaluación (y excluido del entrenamiento) cada vez.

En esta sección clasificaremos a los datos ingresados por medio del método de validación cruzada. Esto nos permitirá realizar comparaciones con el método normal y se podrá comprobar que el método de validación cruzada tiene resultados más estables. A medida que varía K, los valores de precisión varían con mayor suavidad que si validamos con sólo un conjunto de test

### Paso 1: Carga y preparación de los datos.
* **1.1**: Cargar archivo **".csv" o ".txt"** con los datos que se usarán para analizar los modelos KNN que se construirán. El formato del archivo debe ser: valorX,valorY,valorClase. Teniendo en este ejemplo, la coma como separador. También se debe añadir nombre a las columnas.
* **1.2**: Indicar el **separador** de campos del archivo (";", ",", "Tab").
* **1.3**: Se indica que los datos fueron cargados con **éxito**. Se puede comprobar esto haciendo click en la casilla **"Ver Datos"**. También se indica cuántos puntos (filas) tenía el archivo cargado.
* **1.4**: Seleccionar la cantidad de conjuntos a dividir el dataset (**k-fold**). Esta cantidad sólo puede ser 4, 5 o 10 para mayor simplicidad.
* **1.5**: Seleccionar la **proporción** de los datos a **entrenar** (0.7, 0.75, 0.8 y 0.85 son las opciones disponibles,los puntos restantes se usarán para validar los modelos).
* **1.5**: Elegir el **seed** o **semilla** que se usará para mezclar los datos y así obtener los dos conjuntos (training y test). Esto se pide para tener en todo momento de ejecución los mismos puntos de training y test. La forma de obtenerlos es mezclando los datos aleatoriamente y luego cortando y dividiendo justo en la proporción antes indicada. El seed o semilla sólo puede variar entre 1 y 50 para mayor simplicidad.

### Paso 2: Visualización y comparación de los resultados obtenidos
* **2.1**: Se mostrará el grafico de "Valor K VS Coherencias", estadísticos del conjunto de valores de coherencia obtenidos, y un histograma y diagrama de caja del conjunto de valores de coherencia obtenidos, tanto del método de validación cruzada como el método de validación normal.
* **2.2**: Se podrá visualizar una conclusión con información descriptiva de los datos presentados en el paso anterior. 


## Matrices de Confusión
Aquí intentaremos analizar el k óptimo pero incorporando más métricas para el análisis. Estás métricas se obtienen a partir de la matriz de confusión. Esta matriz nos permite distinguir con mayor claridad los puntos testeados como:

* **Verdaderos positivos (VP):** Puntos cuya etiqueta real era 1 y su predicción fue 1.
* **Verdaderos negativos (VN):** Puntos cuya etiqueta real era 0 y su predicción fue 0.
* **Falsos positivos (FP):** Puntos cuya etiqueta real era 0 y su predicción fue 1.
* **Falsos negativos (FN):** Puntos cuya etiqueta real era 1 y su predicción fue 0.

### Paso 1: Carga y preparación de los datos.
* **1.1**: Cargar archivo **".csv" o ".txt"** con los datos que se usarán para analizar los modelos KNN que se construirán. El formato del archivo debe ser: valorX,valorY,valorClase. Teniendo en este ejemplo, la coma como separador. También se debe añadir nombre a las columnas.
* **1.2**: Indicar el **separador** de campos del archivo (";", ",", "Tab").
* **1.3**: Se indica que los datos fueron cargados con **éxito**. Se puede comprobar esto haciendo click en la casilla **"Ver Datos"**. También se indica cuántos puntos (filas) tenía el archivo cargado.
* **1.4**: Seleccionar la **proporción** de los datos a **entrenar** (0.7, 0.75, 0.8 y 0.85 son las opciones disponibles, los puntos restantes se usarán para validar los modelos).
* **1.5**: Elegir el **seed** o **semilla** que se usará para mezclar los datos y así obtener los dos conjuntos (training y test). Esto se pide para tener en todo momento de ejecución los mismos puntos de training y test. La forma de obtenerlos es mezclando los datos aleatoriamente y luego cortando y dividiendo justo en la proporción antes indicada. El seed o semilla sólo puede variar entre 1 y 50 para mayor simplicidad.

### Paso 2: Visualización de los datos obtenidos para cada K
* **2.1**: Se podrá visualizar para cada K, la/s matriz/ces de confusión, distintas métricas obtenidas y, además, existe la posibilidad de seleccionar "Ver curva ROC" para desplegar el gráfico de ROC (Receiver Operating Characteristics).

### Paso 3: Visualización del gráfico de todas las metricas con k de 1 a 25
* **2.1**: Desplazarse al final de la página
* **2.2**: Seleccionar las métricas que se desean visualizar. Deben ser al menos 3.
* **2.3**: Visualizar el gráfico con las métricas seleccionadas, pudiendo variar las mismas y desplazarse por el gráfico. 

