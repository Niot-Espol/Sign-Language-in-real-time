"""Diagnóstico reproducible antes de iniciar una sesión de captura."""

from __future__ import annotations

import csv
from pathlib import Path
import platform

import cv2
import mediapipe as mp

from src.data_collection.catalogo import cargar_catalogo
from src.data_collection.esquema import columnas_dataset


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"MediaPipe: {mp.__version__}")
    catalogo = cargar_catalogo()
    print(f"Catálogo: {len(catalogo)}/20 señas")

    pendientes = [s.etiqueta for s in catalogo if s.tipo == "por_definir" or s.manos_requeridas is None]
    referencias = [s.etiqueta for s in catalogo if not (Path("docs/referencias") / f"{s.slug}.jpg").exists()]
    print(f"Configuración pendiente: {len(pendientes)}")
    print(f"Referencias JPG pendientes: {len(referencias)}")

    esperado = columnas_dataset()
    invalidos = []
    for ruta in Path("data/raw").glob("*.csv"):
        with ruta.open("r", newline="", encoding="utf-8") as archivo:
            encabezado = next(csv.reader(archivo), [])
        if encabezado != esperado:
            invalidos.append(ruta.name)
    print(f"CSV con esquema inválido: {len(invalidos)}")
    if invalidos:
        print("  " + "\n  ".join(invalidos))

    if pendientes:
        print("AVISO: validar tipo y manos requeridas antes de la captura definitiva.")
    elif referencias:
        print("AVISO: faltan fotografías de referencia.")
    elif invalidos:
        print("AVISO: mueve los CSV incompatibles fuera de data/raw.")
    else:
        print("ENTORNO LISTO PARA CAPTURA.")


if __name__ == "__main__":
    main()

