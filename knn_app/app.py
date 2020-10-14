from knn import *
import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)


"""
# Trabajo Final - Inteligencia Artificial 2020 - K-Nearest Neighbors

## Paso 1: Cargá tus datos y seleccioná la proporción de estos para el entrenamiento

"""

file = st.file_uploader('Cargá el archivo con los datos')
if file is not None:
    sep = st.selectbox('Selecciona el separador de campos del archivo',[None, ",", ";", "Tab"])
    if sep is not None:
        data = pd.read_csv(file, sep=sep)

        labels = data.iloc[:,2].unique()
        dictionary = {}
        for i in range(len(labels)):
            dictionary[labels[i]] = np.float(i+1)

        st.write('Los datos fueron cargados **CORRECTAMENTE** :heavy_check_mark:')
        

        if st.checkbox('Ver datos'):
            #fig = plt.figure(figsize=(10,10))
            #observation_colormap = ListedColormap (["red","green","blue","darkorange","purple"])
            #plt.scatter(data[:,0:1], data[:,1:2])
            st.dataframe(data)
            #st.pyplot(fig)
        data = data.replace(to_replace = dictionary)
        data = data.to_numpy()

        
        
        st.write("Hay ", data.shape[0], ' puntos cargados')
        train_prop = st.selectbox('Seleccione la proporción de datos de entrenamiento',
                        [None,0.7,0.75,0.8,0.85]
        )

        if train_prop is not None:

            seed = st.select_slider('Seleccione un valor de seed para mezclar los datos y dividir el conjunto.', list(range(1,50)))
            
            training, testing = split(data, train_prop, round(1-train_prop,2), seed=seed)
            st.write('La proporción de entrenamiento elegida es de: ', train_prop,'la de test es: ', round(1-train_prop,2))
            st.write('El seed elegido para mezclar los datos es de: ', seed)
            """
            ## Paso 2: Con los datos cargados se entrenarán modelos K-nearest Neighbors con K de 1 a N.

            """
            
            n = st.selectbox(
                'Selecciona el valor de N',
                [None] + list(range(1,len(data)))
            )
            if n is not None:

                coherences_multiple_k = test_multiple_knn(training, testing, lastk=n)

                multiple_k_graph = plot_multiple_k(coherences_multiple_k)

                st.write(multiple_k_graph)
                """
                ## Paso 3: Se mostrará el mejor valor de K encontrado y el Grid de predicciones generado con este valor de K.
                
                """
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
                
                best_k = bests_k[0]

                if len(bests_k) != 1:
                    st.write('Los modelos que llegan al máximo valor de coherencia tienen K = ', cad)  
                else:
                    st.write('El modelo con mayor valor de coherencia tiene K = ', best_k)
                
                best_k = st.selectbox('Seleccione el mejor K para obtener su Grid de predicción', bests_k)

                rta = st.radio(
                    'Seleccione el formato del gráfico',
                    ("Normal","Con valores de confianza"), key=1
                )
                plot_format = 'normal' if rta == 'Normal' else 'confidence'

                xx, yy, prediction_grid, confidence_grid, grid_graph = knn_prediction_grid(data, k=best_k, h=0.2, plot = True, plot_format=plot_format)

                st.write(grid_graph)

                """
                ## Paso 4: Seleccioná cualquier K para comparar los Grids.
                
                """
                
                k_selected = st.selectbox(
                    'Selecciona el valor de K',
                    [None] + list(range(1,n+1))
                )

                if k_selected is not None:

                    rta2 = st.radio(
                        'Seleccione el formato del gráfico',
                        ("Normal","Con valores de confianza"), key=2
                    )

                    plot_format2 = 'normal' if rta2 == 'Normal' else 'confidence'

                    'Tu valor seleccionado de K es:', k_selected
                    'El valor de coherencia para K=',k_selected,' es de ', coherences_multiple_k[k_selected-1]

                    xx2, yy2, prediction_grid2, confidence_grid2, grid_graph2 = knn_prediction_grid(data, k=k_selected, h=0.2, plot = True, plot_format=plot_format2)

                    st.write(grid_graph2)

                    """
                    ## Paso 5 - Cargá las coordenadas de nuevos puntos que quieras predecir.
                    * Acá se mostrará la predicción del/ de los punto/s cargado/s y la confianza de esa/s predicción/es con:
                        * El modelo del mejor K (Paso 3) y, 
                        * El modelo del K seleccionado (Paso 4).
                    """
                    