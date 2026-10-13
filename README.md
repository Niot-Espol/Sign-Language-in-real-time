# Traductor de lenguaje de señas en tiempo real

Sistema en Python que usa OpenCV y MediaPipe para capturar los 21 puntos clave de hasta dos manos y clasificarlos mediante un modelo de aprendizaje automático. La predicción se muestra en una interfaz gráfica en tiempo real.

## Instalación

Se recomienda Python 3.10, 3.11 o 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Capturar datos

Ejecuta el recolector indicando el nombre del integrante. Por defecto recorrerá `comer`, `papá`, `ir` y `bien`. Verifica primero que los 21 landmarks estén estables; mantén `Espacio` para guardar una ráfaga limitada a 30 FPS, pulsa `E` para guardar una evidencia y `Q` para salir.

```powershell
python -m src.data_collection.recolector_csv geanella --objetivo 100
```

También existe un panel para integrantes que no quieran usar argumentos de terminal:

```powershell
python -m src.data_collection.app_captura
```

Cada integrante puede comprobar versiones, catálogo, referencias y formato de sus CSV con:

```powershell
python -m src.data_collection.verificar_entorno
```

El catálogo común está en `config/catalogo.json`. Antes de iniciar las capturas definitivas, el equipo debe validar para cada palabra si es estática o dinámica y si requiere una o dos manos. Las dinámicas se capturan como secuencias de 30 frames:

```powershell
python -m src.data_collection.recolector_secuencias geanella "por favor"
```

Se crearán archivos como `data/raw/bien_geanella.csv`. Cada fila contiene 126 características: 63 de la mano izquierda y 63 de la derecha, siempre en ese orden, más la etiqueta como última columna. Las coordenadas se trasladan respecto a cada muñeca y se normalizan por escala. Si una seña utiliza una sola mano, los 63 valores de la mano ausente quedan en cero. El rango admitido es de 80 a 100 muestras por palabra.

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

Antes del entrenamiento, valida y consolida todos los aportes:

```powershell
python -m src.data_collection.consolidar_dataset
```

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) y `docs/protocolo_captura.md` para el flujo compartido entre ramas.

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
