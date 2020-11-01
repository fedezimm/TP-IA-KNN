import streamlit as st
import codecs
from app_steps import *
from test_steps import *

def main():
    
    # Agregamos un selector para el modo de la aplicación en la barra lateral
    st.sidebar.title("TP Final - Grupo 5 - IA")
    mode = st.sidebar.selectbox(
        "Elige el modo de la aplicación",
        ["Modo App", "Modo Tests"]
    )
    
    if mode == "Modo App":
        run_the_app()
    elif mode == "Modo Tests":
        run_tests()

def get_file_content_as_string(path):
    """Función para leer un archivo"""
    f = codecs.open(path,encoding='utf-8')
    file = f.read()
    f.close()
    return file

def run_the_app():

    # Renderizamos el encabezado de la aplicación con st.markdown
    readme_text = st.markdown(get_file_content_as_string("instrucciones_app.md"))
    #Código AppMode
    app_mode = st.sidebar.selectbox(
        "¿Ver Instrucciones o ejecutar aplicación?",
        ["Mostrar instrucciones", "Aplicación"]
    )
    if app_mode == "Mostrar instrucciones":
        st.sidebar.success("Para continuar selecciona la opción 'Aplicación'.")
    elif app_mode == "Aplicación":
        readme_text.empty()
        st.markdown("# Trabajo Final - Grupo 5 - Inteligencia Artificial 2020 - K-Nearest Neighbors")
        step1()

def run_tests():
    # Renderizamos el encabezado de la aplicación con st.markdown
    readme_text = st.markdown(get_file_content_as_string("instrucciones_test.md"))
    #Codigo TestMode
    test_mode = st.sidebar.selectbox(
        "Elige el test",
        ["Mostrar instrucciones","¿Qué metodo de validación elegir?", "Matrices de Confusión"]
    )
    if test_mode == "Mostrar instrucciones":
        st.sidebar.success("Para continuar seleccione algun test.")
    elif test_mode == "¿Qué metodo de validación elegir?":
        readme_text.empty()
        st.markdown("# Trabajo Final - Grupo 5 - Inteligencia Artificial 2020 - K-Nearest Neighbors")
        test1()
    elif test_mode == "Matrices de Confusión":
        readme_text.empty()
        st.markdown("# Trabajo Final - Grupo 5 - Inteligencia Artificial 2020 - K-Nearest Neighbors")
        test2()

main()
