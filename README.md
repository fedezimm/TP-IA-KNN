# TP-IA-KNN

Trabajo práctico integrador de Inteligencia Artificial - 5to año - ISI - UTN-FRRe - 2020

## ATTENTION

This project is based in python 3.7, please make sure that you have this version.
We recommend you to use Windows.

## How to run the project?

### **Option 1**

1. Make sure you have python (version 3.7) and pip or pip3 installed and added to the PATH. You can do it with:

- For more information go to: https://www.python.org/downloads/

2. Clone the repository
   >
   ```
   git clone https://github.com/fedezimm/TP-IA-KNN.git
   ```
3. Execute the following command and make sure that you are in the <b>root folder</b> of the repository
   >
   ```
   cd TP-IA-KNN
   ```
4. Install all dependencies:
   > pip:
   ```
   sudo pip install -r requirements.txt
   ```
   > pip3:
   ```
   sudo pip3 install -r requirements.txt
   ```
5. Now run the app:
   ```
   cd knn_app && streamlit run app.py
   ```
6. Enjoy it!!! 💻💻

#### **Some exceptions**

- If you did all the above steps and you run:

  ```
  streamlit
  ```

  and the message is: <i>'command not found'</i>.
  Then, the problem is that streamlit is not in the path. So, you have to go to the folder where Python is installed and run the command from this folder:

  > Windows:

  ```
  /Python/scripts/streamlit run app.py
  ```

  > Linux or Mac:

  ```
  /usr/local/bin/streamlit run app.py
  ```

- If you try to run the app and you have the following problem with <b>streamlit</b>:
  ```
  Traceback (most recent call last):
     File "usr/local/bin/streamlit", line 5, in <module>
        from streamlit.cli import main
     File "usr/local/bin/streamlit", line 5, in <module>
        from streamlit.cli import main
     ...
     ...
  ```
  Run the followind command:
  > pip:
  ```
  pip install --upgrade protobuf
  ```
  > pip3:
  ```
  pip3 install --upgrade protobuf
  ```

### **Option 2**

1. If you don't want to run locally, just visit: https://knn-ia-g5.herokuapp.com/

There is also a **notebook** where we have developed and documented the python functions that we have used.
