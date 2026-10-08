"""Captura muestras de una seña desde la cámara y las guarda como CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from src.vision.detector_manos import DetectorManos
from src.vision.utilidades_dibujo import dibujar_etiqueta, dibujar_mano


def recolectar(etiqueta: str, salida: Path, camara: int = 0) -> None:
    salida.parent.mkdir(parents=True, exist_ok=True)
    columnas = [f"{eje}{i}" for i in range(21) for eje in ("x", "y", "z")]
    archivo_nuevo = not salida.exists() or salida.stat().st_size == 0
    captura = cv2.VideoCapture(camara)
    if not captura.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    total = 0
    try:
        with salida.open("a", newline="", encoding="utf-8") as archivo, DetectorManos() as detector:
            escritor = csv.writer(archivo)
            if archivo_nuevo:
                escritor.writerow(["label", *columnas])

            while True:
                ok, frame = captura.read()
                if not ok:
                    break
                manos = detector.procesar(frame)
                for mano in manos:
                    dibujar_mano(frame, mano.landmarks)
                dibujar_etiqueta(frame, f"{etiqueta}: {total} | ESPACIO guardar | Q salir")
                cv2.imshow("Recolector de senas", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == 32 and manos:
                    escritor.writerow([etiqueta, *manos[0].caracteristicas.tolist()])
                    archivo.flush()
                    total += 1
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etiqueta", help="Nombre de la seña, por ejemplo: Hola")
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()
    salida = args.salida or Path("data/raw") / f"{args.etiqueta}.csv"
    recolectar(args.etiqueta, salida, args.camara)


if __name__ == "__main__":
    main()

