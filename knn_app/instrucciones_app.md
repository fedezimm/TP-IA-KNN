# Trabajo Final - Grupo 5 - Inteligencia Artificial 2020 - K-Nearest Neighbors

## Aplicación

### Paso 1: Carga y preparación de los datos.
* **1.1**: Cargar archivo **".csv" o ".txt"** con los datos que se usarán para analizar los modelos KNN que se construirán. El formato del archivo debe ser: valorX,valorY,valorClase. Teniendo en este ejemplo, la coma como separador. También se debe añadir nombre a las columnas.
* **1.2**: Indicar el **separador** de campos del archivo (";", ",", "Tab").
* **1.3**: Se indica que los datos fueron cargados con **éxito**. Se puede comprobar esto haciendo click en la casilla **"Ver Datos"**. También se indica cuántos puntos (filas) tenía el archivo cargado.
* **1.4**: Seleccionar la **proporción** de los datos a **entrenar** (0.7, 0.75, 0.8 y 0.85 son las opciones disponibles,los puntos restantes se usarán para validar los modelos).
* **1.5**: Elegir el **seed** o **semilla** que se usará para mezclar los datos y así obtener los dos conjuntos (training y test). Esto se pide para tener en todo momento de ejecución los mismos puntos de training y test. La forma de obtenerlos es mezclando los datos aleatoriamente y luego cortando y dividiendo justo en la proporción antes indicada. El seed o semilla sólo puede variar entre 1 y 50 para mayor simplicidad.
* **1.6**: Se confirmará el valor de seed seleccionado y se podrá tildar la casilla **"Ver distribución de los conjuntos"** para poder analizar las distribuciones de los conjuntos de test y training, para así comprobar que se comportan de forma similar. Si esto no es así, se podrá variar el valor de seed, hasta que se encuentre proporciones deseadas.

### Paso 2: Evaluar la precisión de los modelos según K crezca.
* **2.1**: Seleccionar valor de **N**. Este valor será el **límite de K** hasta el cual se calcularán modelos. Es decir se calcularán modelos KNN con K de 1 a N (parámetro a seleccionar). Este valor de N puede ser hasta la cantidad total de datos menos 1.
* **2.2**: Se mostrará el **gráfico** de las precisiones o valores de coherencia de los modelos con K variando de 1 a N. Este es un **gráfico de línea** con **K** en el **eje X** y la **precisión** del modelo en el **eje Y**.

### Paso 3: Evaluar los mejores modelos KNN obtenidos y sus grillas.
* **3.1**: Se informará la **máxima precisión alcanzada** y el **valor** o los **valores de K** para el/los cual/es se alcanza esa precisión.
* **3.2**: Seleccionar uno de los mejores valores de K encontrados (en el caso de que haya más de uno).
* **3.3**: Seleccionar qué opción de gráfico se quiere ver. **Normal** (grilla normal, la cual es preferible porque lleva menos tiempo) o **Con valores de confianza** (aquí, el gráfico variará de intensidad en los colores según el valor de confianza de las predicciones hechas por el modelo, cuanto más confianza haya en la predicción, más oscuro será el color).
* **3.4**: Seleccionar la **resolución** del gráfico. En este caso se refiere al valor de **"Step"** o **"Paso"** de la grilla. Cuanto más chico, mayor resolución habrá (se generarán celdas más pequeñas). Las opciones disponibles son 0.1, 0.25 o 0.5. Para un gráfico con valores de confianza se **recomienda** elegir 0.25 o 0.5. De otra manera, se tardará mucho en mostrar el gráfico.
* **3.5**: Aquí se muestra la **grilla** para los parámetros pasados. Los colores del fondo indican la predicción del modelo segun los datos de entrenamiento. Los puntos redondos son los datos de entrenamiento del modelo. Los puntos con forma de x son los puntos de validación del modelo.

### Paso 4: Evaluar una grilla con un valor distinto de K.
* **4.1**: Seleccionar un valor de K requerido. Con este valor se generará otra **grilla** a mostrar. Este paso, está pensado para **comparar** la grilla anterior (con la mejor precisión) con cualquier otra grilla con un valor de K distinto. Se podría aquí pedir un valor de K que **no tenga buen valor de precisión**, para poder comparar los modelos. Para poder ver qué valor de K no tiene una buena precisión, se podría ir a ver al gráfico del **paso 2**.
* **4.2**: Seleccionar el formato del gráfico.
* **4.3**: Seleccionar la resolución del gráfico. 
* **4.4**: Se informará el valor de coherencia del modelo y se mostrará la grilla solicitada.

### Paso 5: Carga de un nuevo punto a predecir.
* **5.1**: Introducir un valor de coordenada X del nuevo punto.
* **5.2**: Introducir un valro de coordenada Y del nuevo punto.
* **5.4**: Se informa al usuario las predicciones del punto solicitado en ambos modelos. Estos son, los modelos con K solicitados en el paso 3 y 4. Se muestra la clase predicha por los modelos y el valor de confianza en las predicciones.