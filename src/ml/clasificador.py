"""Carga y uso del clasificador entrenado."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


class ClasificadorSenas:
    def __init__(self, ruta_modelo: Path | str, ruta_labels: Path | str) -> None:
        ruta_modelo = Path(ruta_modelo)
        ruta_labels = Path(ruta_labels)
        if not ruta_modelo.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

        self.modelo = joblib.load(ruta_modelo)
        self.labels: dict[str, str] = {}
        if ruta_labels.exists():
            self.labels = json.loads(ruta_labels.read_text(encoding="utf-8"))

    def predecir(self, caracteristicas: np.ndarray) -> tuple[str, float | None]:
        muestra = np.asarray(caracteristicas, dtype=np.float32).reshape(1, -1)
        prediccion = self.modelo.predict(muestra)[0]
        etiqueta = self.labels.get(str(prediccion), str(prediccion))

        confianza = None
        if hasattr(self.modelo, "predict_proba"):
            confianza = float(np.max(self.modelo.predict_proba(muestra)[0]))
        return etiqueta, confianza

