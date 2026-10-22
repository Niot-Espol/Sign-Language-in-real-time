from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data_collection.consolidar_dataset import consolidar

ENTRADA = Path("data/raw")
SALIDA = Path("data/processed/dataset_consolidado.csv")
MODELO = Path("models/modelo_senas.pkl")
MATRIZ = Path("docs/evidencias/matriz_confusion.png")

dataset, metadata = consolidar(ENTRADA, SALIDA)

print("Muestras por palabra:")
print(dataset["label"].value_counts().to_string())

if dataset["label"].nunique() < 2:
    raise ValueError("Se necesitan al menos 2 palabras para entrenar els modelo.")

X = dataset.drop(columns="label").to_numpy()
y = dataset["label"].astype(str).to_numpy()

es_prueba = np.zeros(len(dataset), dtype=bool)
for archivo, grupo in metadata.groupby("archivo"):
    indices = grupo["sample_id"].to_numpy()
    corte = int(len(indices) * 0.8)
    es_prueba[indices[corte:]] = True

X_train, y_train = X[~es_prueba], y[~es_prueba]
X_test, y_test = X[es_prueba], y[es_prueba]
print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
print(f"Precision: {accuracy_score(y_test, pred) * 100:.1f}%")
print(classification_report(y_test, pred, zero_division=0))

clases = list(modelo.classes_)
matriz = confusion_matrix(y_test, pred, labels=clases)
MATRIZ.parent.mkdir(parents=True, exist_ok=True)
figura, eje = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(matriz, display_labels=clases).plot(ax=eje, colorbar=False)
eje.set_title("Matriz de confusion")
plt.tight_layout()
figura.savefig(MATRIZ, dpi=150)
plt.close(figura)

modelo.fit(X, y)
MODELO.parent.mkdir(exist_ok=True)
joblib.dump(modelo, MODELO)
print(f"Modelo guardado en {MODELO}")
print(f"Matriz de confusion guardada en {MATRIZ}")