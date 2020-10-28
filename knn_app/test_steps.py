import streamlit as st
from knn import *

def test1():
    st.markdown("## ¿Qué método de validación elegir?")
    st.markdown('Demostraremos que el método de validación cruzada tiene resultados más estables. A medida que varía K, los valores de precisión varían con menor variabilidad que si validamos con sólo un conjunto de test.')
    
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

                    



