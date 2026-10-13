# Traductor de lenguaje de señas en tiempo real

Sistema en Python que usa OpenCV y MediaPipe para capturar los 21 puntos clave de una mano y clasificarlos mediante un modelo de aprendizaje automático. La predicción se muestra en una interfaz gráfica en tiempo real.

## Instalación

Se recomienda Python 3.10, 3.11 o 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Capturar datos

Ejecuta el recolector indicando el nombre de la seña. Presiona `Espacio` para guardar una muestra y `Q` para salir.

```powershell
python -m src.data_collection.recolector_csv Hola
```

Cada muestra contiene una etiqueta y 63 características: las coordenadas `x`, `y`, `z` de los 21 landmarks, trasladadas respecto a la muñeca y normalizadas por escala.

## Entrenar y ejecutar

Los cuadernos de `notebooks/` están destinados a consolidar `data/raw/*.csv`, explorar el dataset y entrenar el modelo. El artefacto final debe guardarse como `models/modelo_senas.pkl`; `models/labels.json` puede mapear las clases numéricas a texto, por ejemplo:

```json
{
  "0": "Hola",
  "1": "Gracias"
}
```

La aplicación también abre sin modelo para comprobar la cámara y la detección de la mano:

```powershell
python main.py
```

## Estructura

- `data/raw/`: capturas individuales por seña.
- `data/processed/`: dataset maestro consolidado.
- `models/`: modelo entrenado y etiquetas.
- `notebooks/`: exploración, entrenamiento y evaluación.
- `src/vision/`: detección y landmarks.
- `src/data_collection/`: captura de muestras.
- `src/ml/`: inferencia.
- `src/gui/`: interfaz gráfica.
- `docs/`: manual, póster y evidencias.
