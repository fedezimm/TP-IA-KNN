from knn import *
import streamlit as st

c1 = generate_random_class_points(3, 1, 4, 2, 50, 1)
c2 = generate_random_class_points(7, 2, 4, 1, 50, 2)
c3 = generate_random_class_points(8, 3, 0, 1, 50, 3)
c4 = generate_random_class_points(12, 1, 8, 2, 50, 4)

entire_set = join_groups((c1, c2, c3, c4))

"""
# Trabajo Final - Inteligencia Artificial 2020 - K-Nearest Neighbors

### Paso 1: Cargá tus datos.
* Aquí se debería avisar que los datos se cargaron correctamente.

* Quizás estaría bueno acá que el usuario seleccione la proporcion de training y test para dividir el set. Default: 0.8, 0.2
"""
st.sidebar.markdown('### Paso 1')
train_prop = st.sidebar.selectbox('Seleccione la proporción de datos de entrenamiento',
                    [0.7,0.75,0.8,0.85]
)
'La proporción de entrenamiento elegida es de: ', train_prop,'la de test es: ', round(1-train_prop,2)
"""
### Paso 2: Con los datos cargados se entrenarán modelos K-nearest Neighbors con K de 1 a N.
* El dato N podrá ser elegido por el usuario. 

* Mientras más grande sea este valor, mayor perspectiva se tendrá acerca de cómo mejora o empeora el algoritmo a medida que K crece. 

* Como desventaja, mientras más grande sea K, mayor tiempo tardará la función en computarse.

* Como resultado de este paso, se mostrará un gráfico que nos permite observar el nivel de coherencia que toma el modelo, a medida que crece K.
"""
st.sidebar.markdown('### Paso 2')
n = st.sidebar.slider(
    'Selecciona el valor de N',
     1, int(len(entire_set)/2)
)

training, testing = split(entire_set, train_prop, round(1-train_prop,2))


coherences_multiple_k = test_multiple_knn(training, testing, lastk=n)

multiple_k_graph = plot_multiple_k(coherences_multiple_k)

st.write(multiple_k_graph)
"""
### Paso 3: Se mostrará el mejor valor de K encontrado y el Grid de predicciones generado con este valor de K.
* Este gráfico podrá ser de formato normal (sin valores de confianza) o con valores de confianza.

* Esto lo elige el usuario.
"""
st.sidebar.markdown('### Paso 3')
bests_k, best_coherence= determine_best_k(coherences_multiple_k)
'El máximo valor de coherencia encontrado es: ', best_coherence
cad = ''
for i in range(len(bests_k)):
    if i == len(bests_k) - 2:
        cad += str(bests_k[i]) + ' y '
    elif i == len(bests_k) - 1:
        cad += str(bests_k[i]) + '.'
    else:
        cad += str(bests_k[i]) + ', '
'Los modelos que llegan al máximo valor de coherencia tienen K = ', cad

best_k = bests_k[0]


rta = st.sidebar.radio(
    'Seleccione el formato del gráfico',
    ("Normal","Con valores de confianza"), key=1
)
plot_format = 'normal' if rta == 'Normal' else 'confidence'
'Solicitaste un gráfico ', rta.lower()

xx, yy, prediction_grid, confidence_grid, grid_graph = knn_prediction_grid(entire_set, k=best_k, h=0.2, plot = True, plot_format=plot_format)

st.write(grid_graph)


"""
### Paso 4: El usuario seleccionará un valor de K y el formato de gráfico que desee.
* En este paso, el usuario podrá comparar el valor de coherencia y el grid del modelo para cualquier K que desee contra los datos generados del paso 3.
"""
st.sidebar.markdown('### Paso 4')
k_selected = st.sidebar.slider(
    'Selecciona el valor de k',
     1, int(len(entire_set)/2)
)

rta2 = st.sidebar.radio(
    'Seleccione el formato del gráfico',
    ("Normal","Con valores de confianza"), key=2
)

plot_format2 = 'normal' if rta2 == 'Normal' else 'confidence'

'Tu valor seleccionado de K es:', k_selected
'El valor de coherencia para K=',k_selected,' es de ', coherences_multiple_k[k_selected-1]
'Solicitaste un gráfico ', rta2.lower()

xx2, yy2, prediction_grid2, confidence_grid2, grid_graph2 = knn_prediction_grid(entire_set, k=k_selected, h=0.2, plot = True, plot_format=plot_format2)

st.write(grid_graph2)

"""
### Paso 5 - Cargá las coordenadas de nuevos puntos que quieras predecir.
* Acá se mostrará la predicción del/ de los punto/s cargado/s y la confianza de esa/s predicción/es con:
    * El modelo del mejor K (Paso 3) y, 
    * El modelo del K seleccionado (Paso 4).
"""
st.sidebar.markdown('### Paso 5')