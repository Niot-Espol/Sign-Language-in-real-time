"""Recolector experimental para señas dinámicas como secuencias de landmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from src.data_collection.catalogo import buscar_sena, cargar_catalogo
from src.data_collection.recolector_csv import _nombre_seguro, _mostrar_referencia
from src.vision.detector_manos import DetectorManos
from src.vision.utilidades_dibujo import dibujar_mano


def recolectar_secuencias(
    integrante: str,
    palabra: str,
    repeticiones: int = 10,
    frames_por_secuencia: int = 30,
    fps: float = 30.0,
    camara: int = 0,
) -> None:
    sena = buscar_sena(palabra, cargar_catalogo())
    integrante = _nombre_seguro(integrante)
    salida = Path("data/raw/sequences") / sena.slug / integrante
    salida.mkdir(parents=True, exist_ok=True)
    captura = cv2.VideoCapture(camara)
    if not captura.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    grabando = False
    cuenta_regresiva: float | None = None
    secuencia: list[np.ndarray] = []
    ultima_muestra = 0.0
    completadas = len(list(salida.glob("*.npz")))

    try:
        with DetectorManos(max_manos=2) as detector:
            while completadas < repeticiones:
                ok, frame = captura.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                manos = detector.procesar(frame)
                for mano in manos:
                    dibujar_mano(frame, mano.landmarks)

                ahora = time.perf_counter()
                if cuenta_regresiva is not None:
                    restante = max(0, int(np.ceil(cuenta_regresiva - ahora)))
                    estado = f"INICIA EN {restante}"
                    if ahora >= cuenta_regresiva:
                        cuenta_regresiva = None
                        grabando = True
                        secuencia = []
                        ultima_muestra = 0.0
                elif grabando:
                    estado = f"GRABANDO {len(secuencia)}/{frames_por_secuencia}"
                else:
                    estado = "ESPACIO: nueva repeticion"

                cv2.putText(
                    frame, f"{sena.etiqueta} | {completadas}/{repeticiones} | {estado}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (40, 220, 40) if grabando else (255, 255, 255), 2, cv2.LINE_AA,
                )
                visor = _mostrar_referencia(frame, sena.slug)
                cv2.imshow("Recolector de secuencias", visor)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == 32 and not grabando and cuenta_regresiva is None:
                    cuenta_regresiva = ahora + 3.0

                cantidad_valida = (
                    bool(manos)
                    and (sena.manos_requeridas is None or len(manos) == sena.manos_requeridas)
                )
                if grabando and cantidad_valida and ahora - ultima_muestra >= 1.0 / fps:
                    secuencia.append(detector.vector_dos_manos(manos))
                    ultima_muestra = ahora
                    if len(secuencia) == frames_por_secuencia:
                        marca = time.strftime("%Y%m%d_%H%M%S")
                        ruta = salida / f"{sena.slug}_{integrante}_{marca}_{completadas + 1:02d}.npz"
                        np.savez_compressed(
                            ruta,
                            frames=np.stack(secuencia),
                            label=sena.etiqueta,
                            integrante=integrante,
                            fps=fps,
                        )
                        completadas += 1
                        grabando = False
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integrante")
    parser.add_argument("palabra")
    parser.add_argument("--repeticiones", type=int, default=10)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()
    recolectar_secuencias(
        args.integrante, args.palabra, args.repeticiones, args.frames, camara=args.camara
    )


if __name__ == "__main__":
    main()

