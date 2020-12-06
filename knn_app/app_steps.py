import streamlit as st
from knn import *

def step1():
    #--------------------------------------------------STEP 1--------------------------------------------------------------
    st.markdown('## Paso 1: Carga y preparación de los datos.')
    #Carga de archivo
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
            #Se reemplazan los labels de los datos por numeros de 0 a N para poder manejar en np.ndarray
            data = data.replace(to_replace = dictionary)
            data = data.to_numpy()
            st.write("Hay ", data.shape[0], ' puntos cargados')
            #Selección de la proporcion de entrenamiento
            train_prop = st.selectbox('Seleccione la proporción de datos de entrenamiento',
                            [0.7,0.75,0.8,0.85]
            )            
            if train_prop is not None:                
                st.write('La proporción de entrenamiento elegida es de: ', train_prop,'la de test es: ', round(1-train_prop,2))
                #Selección del seed
                seed = st.select_slider('Seleccione un valor de seed para mezclar los datos y dividir el conjunto.', list(range(1,51)))                
                #División del conjunto en training y test
                training, testing = split(data, train_prop, round(1-train_prop,2), seed=seed)
                st.write('El seed elegido para mezclar los datos es de: ', seed)
                #Opción de ver la distribución de los conjuntos
                if st.checkbox('Ver Distribución de los conjuntos'):
                    st.write(plot_training_test_distributions(training, testing))
                    st.write("Hay ", len(training), " puntos de entrenamiento y ", len(testing), " puntos de testeo.")
                    #Se ejecuta el paso 2.
                step2(data, training, testing, labels)

def step2(data, training, testing, labels):
    #------------------------------------------------- STEP 2 ----------------------------------------------------------
    st.markdown('## Paso 2: Evaluar la precisión de los modelos según K crezca.')
    # Seleccion del valor de N                
    n = st.selectbox(
        'Selecciona el valor de N',
        [100] + list(range(1,len(training)))
    )
    if n is not None:
        # Se testean los modelos con k de 1 a N. Se usan siempre los mismos datos de test.
        coherences_multiple_k = test_multiple_knn(training, testing, lastk=n)
        # Se grafica K vs Accuracy obtenido
        multiple_k_graph = plot_multiple_k(coherences_multiple_k)[1]
        st.write(multiple_k_graph)
        #Se ejecuta el paso 3.
        step3(coherences_multiple_k, training, testing, n, labels)

def step3(coherences_multiple_k, training, testing, n, labels):
    #--------------------------------------------------------------- STEP 3 ---------------------------------------------------------------
    st.markdown('## Paso 3: Evaluar los mejores modelos KNN obtenidos y sus grillas.')
    # Se determina el/los mejor/es K con el mejor valor de coherencia.
    bests_k, best_coherence= determine_best_k(coherences_multiple_k)
    st.write('El máximo valor de coherencia encontrado es: ', best_coherence)
    #Se arma una cadena para mostrarla. Esta cadena dice los mejores K encontrados en el caso de que haya mas que uno.
    cad = ''
    for i in range(len(bests_k)):
        if i == len(bests_k) - 2:
            cad += str(bests_k[i]) + ' y '
        elif i == len(bests_k) - 1:
            cad += str(bests_k[i]) + '.'
        else:
            cad += str(bests_k[i]) + ', '                               
    #Se extrae el primer valor de la lista para mostrar el mejor K en el caso de que sea uno solo
    best_k = bests_k[0]
    if len(bests_k) != 1:
        st.write('Los modelos que llegan al máximo valor de coherencia tienen K = ', cad)  
    else:
        st.write('El modelo con mayor valor de coherencia tiene K = ', best_k)
    #Seleccion de uno de los mejores valores de K para graficar el grid.                        
    best_k = st.selectbox('Seleccione el mejor K para obtener su Grid de predicción', bests_k)
    #Seleccion del formato del gráfico
    rta = st.radio(
        'Seleccione el formato del gráfico',
        ("Normal","Con valores de confianza"), key=1
    )
    #Seleccion de la resolución del gráfico. 
    h1 = st.selectbox(
        'Seleccione la resolución del gráfico',
        [0.5,0.25,0.1], key=1
    )
    plot_format = 'normal' if rta == 'Normal' else 'confidence'
    if h1 is not None:
        #Generación de la grilla y gráfico.
        xx, yy, prediction_grid, confidence_grid, grid_graph = knn_prediction_grid(training, k=best_k, h=h1, plot = True, plot_format=plot_format, testing=testing, labels=labels)
        st.pyplot(grid_graph)
        #Se ejecuta el paso 4.
        step4(n, training, testing, best_k, coherences_multiple_k, labels)

def step4(n, training, testing, best_k, coherences_multiple_k, labels):
    #----------------------------------------------------------------------- STEP 4 ----------------------------------------------------------
    st.markdown('## Paso 4: Evaluar una grilla con un valor distinto de K.')
    # Selección de K
    k_selected = st.selectbox(
        'Selecciona el valor de K',
        list(range(1,n+1))
    )
    if k_selected is not None:
        # Selección del formato de gráfico
        rta2 = st.radio(
            'Seleccione el formato del gráfico',
            ("Normal","Con valores de confianza"), key=2
        )
        # Selección de la resolución del gráfico
        h2 = st.selectbox(
            'Seleccione la resolución del gráfico',
            [0.5,0.25,0.1], key=2
        )
        if h2 is not None:
            plot_format2 = 'normal' if rta2 == 'Normal' else 'confidence'
            st.write('Tu valor seleccionado de K es:', k_selected)
            st.write('El valor de coherencia para K=',k_selected,' es de ', coherences_multiple_k[k_selected-1])
            # Generación del grid y graficación
            xx2, yy2, prediction_grid2, confidence_grid2, grid_graph2 = knn_prediction_grid(training, k=k_selected, h=h2, plot = True, plot_format=plot_format2, testing=testing, labels=labels)
            st.pyplot(grid_graph2)
            # Se ejecuta el paso 5.
            step5(training, best_k, k_selected, labels)
def step5(training, best_k, k_selected, labels):
    #------------------------------------------------------------- STEP 5 -------------------------------------------------------------- 
    st.markdown('## Paso 5: Carga de un nuevo punto a predecir.')
    # Selección de la coordenada X del punto.
    x_new = st.number_input(
        'Selecciona la coordenada X del nuevo punto a predecir',
        min_value= float(np.min(training[:,0]) - 3.0),
        max_value= float(np.max(training[:,0]) + 3.0),
        value=np.mean(training[:,0])
    )
    
    # Seleccion de la coordenada Y del punto
    y_new = st.number_input(
        'Selecciona la coordenada Y del nuevo punto a predecir',
        min_value= float(np.min(training[:,1]) - 3.0),
        max_value= float(np.max(training[:,1]) + 3.0),
        value=np.mean(training[:,1])
    )
    
    # Generación del nuevo punto
    new_point = np.array([x_new, y_new]).transpose()
    # Predicción con el mejor K seleccionado (paso 3).
    best_predict = knn_predict(training, new_point, best_k)
    # Predicción con el K seleccionado en paso 4.
    predict2 = knn_predict(training, new_point, k_selected)
    st.write('Con K = ',best_k,' el nuevo punto pertenece a la clase ',labels[best_predict[0]-1],', con una confianza de: ',round(best_predict[1],3))
    st.write('Con K = ',k_selected,' el nuevo punto pertenece a la clase ',labels[predict2[0]-1],', con una confianza de: ',round(predict2[1],3))