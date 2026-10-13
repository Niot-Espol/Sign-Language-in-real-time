"""Captura controlada de landmarks para una o varias señas."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import ctypes
import os
from pathlib import Path
import re
import time
import unicodedata

import cv2
import numpy as np

from src.data_collection.catalogo import Sena, buscar_sena, cargar_catalogo
from src.data_collection.esquema import columnas_dataset, columnas_landmarks
from src.vision.detector_manos import DetectorManos
from src.vision.utilidades_dibujo import dibujar_etiqueta, dibujar_mano


PALABRAS_PILOTO = ("comer", "papá", "ir", "bien")
TECLA_ESPACIO_WINDOWS = 0x20
FRAMES_ESTABLES = 8
UMBRAL_ESTABILIDAD = 0.018


def _nombre_seguro(texto: str) -> str:
    """Convierte nombres y etiquetas en componentes seguros para archivos."""
    normalizado = unicodedata.normalize("NFKD", texto.strip().lower())
    sin_tildes = "".join(c for c in normalizado if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9_-]+", "_", sin_tildes).strip("_")


def _filas_existentes(salida: Path) -> int:
    if not salida.exists():
        return 0
    with salida.open("r", newline="", encoding="utf-8") as archivo:
        return max(sum(1 for _ in archivo) - 1, 0)


def _validar_columnas_existentes(salida: Path, columnas: list[str]) -> None:
    """Impide mezclar capturas antiguas de 63 valores con las nuevas de 126."""
    if not salida.exists() or salida.stat().st_size == 0:
        return
    with salida.open("r", newline="", encoding="utf-8") as archivo:
        encabezado = next(csv.reader(archivo), [])
    esperado = columnas_dataset()
    if encabezado != esperado:
        raise ValueError(
            f"{salida} usa un formato anterior de {len(encabezado) - 1} "
            "características. Consérvalo como respaldo y captura nuevamente; "
            "no se pueden mezclar filas de 63 y 126 valores."
        )


def _espacio_presionado(tecla_opencv: int) -> bool:
    """Detecta el estado sostenido de Espacio; OpenCV solo informa pulsaciones."""
    if os.name == "nt":
        return bool(ctypes.windll.user32.GetAsyncKeyState(TECLA_ESPACIO_WINDOWS) & 0x8000)
    return tecla_opencv == 32


def _es_estable(historial: deque[np.ndarray]) -> bool:
    if len(historial) < historial.maxlen:
        return False
    matriz = np.stack(historial)
    columnas_activas = np.any(matriz != 0, axis=0)
    if not np.any(columnas_activas):
        return False
    return float(np.mean(np.std(matriz[:, columnas_activas], axis=0))) < UMBRAL_ESTABILIDAD


def _mostrar_referencia(frame: np.ndarray, slug: str) -> np.ndarray:
    ruta = Path("docs/referencias") / f"{slug}.jpg"
    referencia = cv2.imread(str(ruta)) if ruta.exists() else None
    if referencia is None:
        return frame
    alto = frame.shape[0]
    ancho = max(int(referencia.shape[1] * alto / referencia.shape[0]), 1)
    referencia = cv2.resize(referencia, (ancho, alto))
    cv2.putText(
        referencia, "REFERENCIA", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.hstack((frame, referencia))


def recolectar(
    sena: Sena,
    salida: Path,
    camara: int = 0,
    objetivo: int = 100,
    fps: float = 30.0,
) -> int:
    """Guarda una ráfaga controlada hasta alcanzar ``objetivo`` muestras.

    ESPACIO mantenido activa la ráfaga. E guarda una evidencia del visor y Q sale.
    Se usa un intervalo temporal explícito para no superar los FPS configurados.
    """
    salida.parent.mkdir(parents=True, exist_ok=True)
    columnas = columnas_landmarks()
    _validar_columnas_existentes(salida, columnas)
    archivo_nuevo = not salida.exists() or salida.stat().st_size == 0
    captura = cv2.VideoCapture(camara)
    if not captura.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    total = _filas_existentes(salida)
    capturando = False
    ultimo_guardado = 0.0
    directorio_evidencias = Path("docs/evidencias")
    directorio_evidencias.mkdir(parents=True, exist_ok=True)
    historial: deque[np.ndarray] = deque(maxlen=FRAMES_ESTABLES)
    try:
        with salida.open("a", newline="", encoding="utf-8") as archivo, DetectorManos(max_manos=2) as detector:
            escritor = csv.writer(archivo)
            if archivo_nuevo:
                escritor.writerow(columnas_dataset())

            while total < objetivo:
                ok, frame = captura.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                manos = detector.procesar(frame)
                for mano in manos:
                    dibujar_mano(frame, mano.landmarks)
                vector = detector.vector_dos_manos(manos)
                historial.append(vector)
                cantidad_valida = (
                    sena.manos_requeridas is None
                    or len(manos) == sena.manos_requeridas
                )
                estable = bool(manos) and cantidad_valida and _es_estable(historial)
                estado = "CAPTURANDO" if capturando else "VERIFICAR LANDMARKS"
                dibujar_etiqueta(
                    frame,
                    f"{sena.etiqueta}: {total}/{objetivo} - {estado} - manos: {len(manos)}/2",
                )
                color = (40, 220, 40) if estable else (40, 40, 230)
                cv2.putText(
                    frame,
                    "ESTABLE" if estable else "ESPERE: estabilice manos y landmarks",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame, "ESPACIO capturar | E evidencia | Q salir", (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )
                visor = _mostrar_referencia(frame, sena.slug)
                cv2.imshow("Recolector de senas", visor)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                capturando_antes = capturando
                capturando = _espacio_presionado(tecla)
                if capturando and not capturando_antes:
                    ultimo_guardado = 0.0
                if tecla in (ord("e"), ord("E")):
                    marca = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(directorio_evidencias / f"{sena.slug}_{marca}.jpg"), frame)

                ahora = time.perf_counter()
                if capturando and estable and ahora - ultimo_guardado >= 1.0 / fps:
                    escritor.writerow([*vector.tolist(), sena.etiqueta])
                    archivo.flush()
                    total += 1
                    ultimo_guardado = ahora
    finally:
        captura.release()
        cv2.destroyAllWindows()
    return total


def recolectar_catalogo(
    integrante: str,
    palabras: list[str],
    camara: int,
    objetivo: int,
) -> None:
    integrante_seguro = _nombre_seguro(integrante)
    if not integrante_seguro:
        raise ValueError("El nombre del integrante no puede quedar vacío.")

    catalogo = cargar_catalogo()
    for indice, palabra in enumerate(palabras, start=1):
        sena = buscar_sena(palabra, catalogo)
        salida = Path("data/raw") / f"{sena.slug}_{integrante_seguro}.csv"
        print(f"[{indice}/{len(palabras)}] Preparando '{sena.etiqueta}' -> {salida}")
        if sena.tipo == "por_definir" or sena.manos_requeridas is None:
            print("AVISO: el equipo aún debe validar tipo y cantidad de manos en config/catalogo.json")
        total = recolectar(sena, salida, camara, objetivo)
        print(f"'{sena.etiqueta}': {total}/{objetivo} muestras disponibles.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integrante", help="Nombre de quien captura, por ejemplo: geanella")
    parser.add_argument(
        "--palabras",
        nargs="+",
        default=list(PALABRAS_PILOTO),
        help="Señas a capturar (por defecto: comer papá ir bien)",
    )
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--objetivo", type=int, default=100, choices=range(80, 101), metavar="80-100")
    args = parser.parse_args()
    recolectar_catalogo(args.integrante, args.palabras, args.camara, args.objetivo)


if __name__ == "__main__":
    main()
