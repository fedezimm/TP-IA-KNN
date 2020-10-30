from setuptools import setup
#Project based on Python 3
#First run -python3 setup.py build- then -python3 setup.py install-
setup(
   name='knn_app',
   version='1.0',
   description='A KNN grapher',
   author='Team N5 - IA - UTN',
   author_email=['fedezimm@gmail.com','carlosszaracho@gmail.com','geraenrique97@hotmail.com','nikoalegre97@gmail.com'],
   install_requires=['streamlit','scikit-learn','scipy','matplotlib','pandas','numpy']
 
)
