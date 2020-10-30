import streamlit as st
from knn import *

def test1():
    st.markdown("## ¿Qué método de validación elegir?")
    st.markdown('Demostraremos que el método de validación cruzada tiene resultados más estables. A medida que varía K, los valores de precisión varían con mayor suavidad que si validamos con sólo un conjunto de test.')
    
    file = st.file_uploader('Cargá el archivo con los datos')

    if file is not None:
        file.seek(0)
        #Seleccionde separador
        sep = st.selectbox('Selecciona el separador de campos del archivo',[None, ",", ";", "Tab"])
        if sep is not None:  
            #Carga de los datos en un pd.dataframe
            data = pd.read_csv(file, sep=sep)
            #Se extraen las etiquetas de las clases
            labels = data.iloc[:,2].unique()
            dictionary = {}
            #Se asignan nuevas etiquetas (numeros enteros)
            for i in range(len(labels)):
                dictionary[labels[i]] = np.float(i+1)
            #Se confirma la carga correcta de los datos
            st.write('Los datos fueron cargados **CORRECTAMENTE** :heavy_check_mark:')
            #Se muestran los datos al tildar el checkbox
            if st.checkbox('Ver datos'):
                st.dataframe(data)
            #Se reemplazan los labels de los datos por numeros de 1 a N para poder manejar en np.ndarray
            data = data.replace(to_replace = dictionary)
            data = data.to_numpy()
            cv = st.selectbox(
                'Validación cruzada: Selecciona la cantidad de conjuntos a dividir el dataset (k-fold)',
                [None,4,5,10]
            )
            if cv is not None:
                train_prop = st.selectbox(
                    'Validación Normal: Selecciona la proporción de datos de entrenamiento para dividir el conjunto',
                    [None,0.7,0.75,0.8,0.85]
                )
                if train_prop is not None:
                    seed = st.select_slider(
                        'Seleccione un valor de seed para mezclar los datos y dividir el conjunto.', 
                        list(range(1,51))
                    )
                    k_max = round(len(data)/8)
                    cv_values, cv_graph, cv_metrics = get_cross_validation_metrics(data, k_max=k_max, cv=cv, seed=seed)
                    nv_values, nv_graph, nv_metrics = get_normal_validation_metrics(data, train_prop, seed=seed, k_max=k_max)
                    st.markdown("### Validación cruzada")
                    cad = ''
                    cv_bests = determine_best_k(cv_values)
                    for i in range(len(cv_bests[0])):
                        if i == len(cv_bests[0]) - 2:
                            cad += str(cv_bests[0][i]) + ' y '
                        elif i == len(cv_bests[0]) - 1:
                            cad += str(cv_bests[0][i]) + '.'
                        else:
                            cad += str(cv_bests[0][i]) + ', '
                    st.write("La máxima coherencia es de: " , cv_bests[1] , ". Se alcanza con K = " , cad)
                    st.markdown("#### Gráfico: Valor K VS Coherencias")
                    st.pyplot(cv_graph)
                    st.markdown("#### Estadísticos del conjunto de valores de coherencia obtenidos")
                    st.write(cv_metrics)
                    st.markdown("#### Histograma y Diagrama de caja del conjunto de valores de coherencia obtenidos")
                    st.pyplot(get_histogram_boxplot(cv_values))
                    st.markdown("### Validación Normal")
                    nv_bests = determine_best_k(nv_values)
                    cad = ''
                    for i in range(len(nv_bests[0])):
                        if i == len(nv_bests[0]) - 2:
                            cad += str(nv_bests[0][i]) + ' y '
                        elif i == len(nv_bests[0]) - 1:
                            cad += str(nv_bests[0][i]) + '.'
                        else:
                            cad += str(nv_bests[0][i]) + ', '
                    st.write("La máxima coherencia es de: " , nv_bests[1] , ". Se alcanza con K = " , cad)
                    st.markdown("#### Gráfico: Valor K VS Coherencias")
                    st.pyplot(nv_graph)
                    st.markdown("#### Estadísticos del conjunto de valores de coherencia obtenidos")
                    st.write(nv_metrics)
                    st.markdown("#### Histograma y Diagrama de caja del conjunto de valores de coherencia obtenidos")
                    st.pyplot(get_histogram_boxplot(nv_values))

                    st.markdown("### Conclusión")
                    st.markdown("Tanto los **gráficos** (de líneas, diagrama de caja e histograma) como los **estadísticos** (varianza, desvío estándar y percentiles) nos demuestran que hay una mayor dispersión en los valores de precisión del algoritmo cuando validamos de forma **tradicional**.")
                    
                    st.markdown("Esto nos lleva a concluir que la **validación cruzada** nos proporciona valores de precisión que varían moderadamente a medida que K cambia, por lo tanto sus resultados son más fiables.")

                    st.markdown("Si se observa:")
                    st.markdown("   * **Gráfico de líneas**: Validación cruzada presenta una línea que se parece mayormente a una curva, comparándola con la presentada por la validación normal, la cual presenta una línea más escalonada.")
                    st.markdown("   * **Estadísticos**: Los valores de varianza y desvío estándar son menores en la validación cruzada, lo que nos indica que hay menos dispersión en los datos de precisión. También, generalmente, el rango de valores está más acotado y los percentiles están más juntos.")
                    st.markdown("   * **Histograma**: Aquí se puede ver que en Validación cruzada los datos se encuentran máyormente agrupadados que los datos de la validación normal.")
                    st.markdown("   * **Diagrama de caja**: Aquí se puede ver con mayor facilidad la dispersión de los datos. Generalmente la caja es más pequeña para Validación cruzada")     
                    
def test2():
    st.markdown("## Evaluación del mejor K analizando otras métricas obtenidas a partir de la Matriz de Confusión")
    st.markdown("Aquí intentaremos analizar el k óptimo pero incorporando más métricas para el análisis. Estás métricas se obtienen a partir de la **matriz de confusión**. Esta matriz nos permite distinguir con mayor claridad los puntos testeados como:")
    st.markdown("   * **Verdaderos positivos (VP)**: Puntos cuya etiqueta real era 1 y su predicción fue 1.")
    st.markdown("   * **Verdaderos negativos (VN)**: Puntos cuya etiqueta real era 0 y su predicción fue 0.")
    st.markdown("   * **Falsos positivos (FP)**: Puntos cuya etiqueta real era 0 y su predicción fue 1.")
    st.markdown("   * **Falsos negativos (FN)**: Puntos cuya etiqueta real era 1 y su predicción fue 0.")
    
    file = st.file_uploader('Cargá el archivo con los datos')

    if file is not None:
        file.seek(0)
        #Seleccionde separador
        sep = st.selectbox('Selecciona el separador de campos del archivo',[None, ",", ";", "Tab"])
        if sep is not None:  
            #Carga de los datos en un pd.dataframe
            data = pd.read_csv(file, sep=sep)
            #Se extraen las etiquetas de las clases
            labels = data.iloc[:,2].unique()
            dictionary = {}
            #Se asignan nuevas etiquetas (numeros enteros)
            for i in range(len(labels)):
                dictionary[labels[i]] = np.float(i+1)
            #Se confirma la carga correcta de los datos
            st.write('Los datos fueron cargados **CORRECTAMENTE** :heavy_check_mark:')
            #Se muestran los datos al tildar el checkbox
            if st.checkbox('Ver datos'):
                st.dataframe(data)
            #Se reemplazan los labels de los datos por numeros de 1 a N para poder manejar en np.ndarray
            data = data.replace(to_replace = dictionary)
            data = data.to_numpy()

            train_prop = st.selectbox(
                    'Selecciona la proporción de datos de entrenamiento para dividir el conjunto',
                    [None,0.7,0.75,0.8,0.85]
                )
            
            if train_prop is not None:
                seed = st.select_slider('Seleccione un valor de seed para mezclar los datos y dividir el conjunto.', list(range(1,51)))
                training, test = split(data, train_prop, 1-train_prop, seed)
                if len(labels) == 2:
                    index = []
                    columns = []
                    for j in range(len(labels)):
                        cadInd = "Valor actual " + str(labels[j])
                        cadCol = "Prediccion " + str(labels[j])
                        index.append(cadInd)
                        columns.append(cadCol)
                    roc_aucs = []
                    for i in range(1,26):
                        confusion_matrix = get_confusion_matrix_binary(training, test, k=i, threshold=None)
                        mat = pd.DataFrame(confusion_matrix, index=index,columns=columns)
                        st.markdown("### K = "+str(i))
                        st.dataframe(mat.style.highlight_max(axis=0))
                        st.write(get_cm_metrics(confusion_matrix))
                        roc, fprs, tprs, roc_auc, threshold = plot_binary_classification_roc_metric(training, test, k=i)
                        roc_aucs.append(roc_auc)
                        if st.checkbox('Ver Curva ROC',key=i):
                            st.pyplot(roc)
                    
                    m = plot_cm_metrics_list(get_list_of_metrics(training, test, k_max = 25))
                    st.markdown("### Gráfico de todas las metricas con k de 1 a 25")
                    choices = ["Sensibilidad","Especificidad","Precisión","Valor de predicción negativo","Tasa de pérdida","Tasa de caída","Tasa de descubrimiento falso","Tasa de omisiones falsas","Coherencia","Coherencia balanceada","F-score"]
                    selected = st.multiselect(
                        "Seleccione las métricas que quiere visualizar",
                        choices
                    )
                    if len(selected) > 2:
                        array = np.array([m[1][option] for option in selected]).transpose()
                        chart_data = pd.DataFrame(
                            array,
                            index=m[0],
                            columns=selected
                        )
                        st.line_chart(chart_data, width=300, height=600)
                else:
                    for i in range(1,26):
                        confusion_matrixes = get_classes_confusion_matrixes(training, test, i)
                        st.markdown("### K = "+str(i))
                        for j in range(len(confusion_matrixes)):
                            c = labels[j]
                            st.markdown("#### Clase = "+str(c))
                            mat = pd.DataFrame(confusion_matrixes[j], index=["Valor actual 1","Valor actual 0"],columns=["Prediccion 1","Prediccion 0"])
                            st.dataframe(mat.style.highlight_max(axis=0))
                        st.write(get_cm_metrics_multiple_classes(confusion_matrixes))
                        roc = plot_multiple_classification_roc_metric(training, test, labels, k=i)[0]
                        if st.checkbox('Ver Curvas ROC',key=i):
                            st.pyplot(roc)
                    m = plot_cm_metrics_list(get_list_of_metrics_mult(training, test, k_max = 25))
                    st.markdown("### Gráfico de todas las metricas con k de 1 a 25")
                    choices = ["Sensibilidad","Especificidad","Precisión","Valor de predicción negativo","Tasa de pérdida","Tasa de caída","Tasa de descubrimiento falso","Tasa de omisiones falsas","Coherencia","Coherencia balanceada","F-score"]
                    selected = st.multiselect(
                        "Seleccione las métricas que quiere visualizar",
                        choices
                    )
                    if len(selected) > 2:
                        array = np.array([m[1][option] for option in selected]).transpose()
                        chart_data = pd.DataFrame(
                            array,
                            index=m[0],
                            columns=selected
                        )
                        st.line_chart(chart_data, width=300, height=600)





