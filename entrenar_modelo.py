import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Definir las rutas y las señas que entrenaremos
carpeta_evidencias = "docs/evidencias"
señas = ["querer", "tu"]

X = []  # Aquí guardaremos las coordenadas
Y = []  # Aquí guardaremos las etiquetas (el nombre de la seña)

# 2. Leer los datos de cada archivo CSV
for seña in señas:
    ruta_csv = os.path.join(carpeta_evidencias, f"{seña}.csv")
    
    if os.path.exists(ruta_csv):
        print(f"Leyendo datos para la seña: {seña}")
        df = pd.read_csv(ruta_csv)
        
        # Convertimos las filas a listas de números
        for index, row in df.iterrows():
            X.append(row.tolist())
            Y.append(seña)
    else:
        print(f"Advertencia: No se encontró el archivo {ruta_csv}")

if len(X) == 0:
    print("Error: No se encontraron datos para entrenar. Asegúrate de haber grabado las muestras.")
else:
    # 3. Dividir los datos para entrenamiento y prueba (80% entrenar, 20% probar)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 4. Crear y entrenar el modelo (Random Forest)
    print("Entrenando el modelo...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 5. Evaluar la precisión
    precision = modelo.score(X_test, y_test)
    print(f"¡Modelo entrenado con éxito! Precisión en pruebas: {precision * 100:.2f}%")

    # 6. Guardar el modelo entrenado en la carpeta 'models/'
    os.makedirs("models", exist_ok=True)
    ruta_modelo = "models/modelo_señas.pkl"
    
    with open(ruta_modelo, "wb") as f:
        pickle.dump(modelo, f)
        
    print(f"Modelo guardado correctamente en: {ruta_modelo}")