from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

archivos = sorted(Path("data/raw").glob("*.csv"))
if not archivos:
    raise FileNotFoundError("No hay archivos en data/raw")

datos = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
datos = datos.dropna()

Path("data/processed").mkdir(parents=True, exist_ok=True)
datos.to_csv("data/processed/dataset_consolidado.csv", index=False)

X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].astype(str).values

print(datos.iloc[:, -1].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
print("Precision:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

Path("models").mkdir(exist_ok=True)
joblib.dump(modelo, "models/modelo_senas.pkl")
print("Modelo guardado en models/modelo_senas.pkl")