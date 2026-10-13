"""Detección de manos y extracción de características con MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class ResultadoMano:
    """Landmarks y características normalizadas de una mano."""

    landmarks: object
    puntos: np.ndarray
    caracteristicas: np.ndarray
    lateralidad: str


class DetectorManos:
    """Pequeño envoltorio de MediaPipe Hands para cámara y entrenamiento."""

    def __init__(
        self,
        max_manos: int = 2,
        confianza_deteccion: float = 0.6,
        confianza_seguimiento: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._detector = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_manos,
            min_detection_confidence=confianza_deteccion,
            min_tracking_confidence=confianza_seguimiento,
        )

    def procesar(self, frame_bgr: np.ndarray) -> list[ResultadoMano]:
        """Devuelve hasta dos manos detectadas y 63 valores por mano."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resultado = self._detector.process(frame_rgb)
        if not resultado.multi_hand_landmarks:
            return []

        lateralidades = resultado.multi_handedness or []
        manos: list[ResultadoMano] = []
        for indice, landmarks in enumerate(resultado.multi_hand_landmarks):
            etiqueta = "Desconocida"
            if indice < len(lateralidades):
                etiqueta = lateralidades[indice].classification[0].label
            manos.append(
                ResultadoMano(
                    landmarks=landmarks,
                    puntos=self._puntos(landmarks),
                    caracteristicas=self._normalizar(landmarks),
                    lateralidad=etiqueta,
                )
            )
        return manos

    @staticmethod
    def vector_dos_manos(manos: list[ResultadoMano]) -> np.ndarray:
        """Crea 126 valores fijos: 63 de izquierda y 63 de derecha.

        La mitad correspondiente a una mano ausente se completa con ceros. Este
        tamaño fijo permite mezclar señas de una y dos manos en el mismo modelo.
        """
        vacia = np.zeros((21, 3), dtype=np.float32)
        por_lado: dict[str, ResultadoMano] = {}
        desconocidas: list[ResultadoMano] = []

        for mano in manos:
            lado = mano.lateralidad.lower()
            if lado in ("left", "right") and lado not in por_lado:
                por_lado[lado] = mano
            else:
                desconocidas.append(mano)

        # Respaldo para casos raros en los que MediaPipe no entregue lateralidad.
        for mano in desconocidas:
            lado_libre = "left" if "left" not in por_lado else "right"
            if lado_libre not in por_lado:
                por_lado[lado_libre] = mano

        presentes = list(por_lado.values())
        if not presentes:
            return np.zeros(126, dtype=np.float32)

        # El origen y la escala comunes conservan la relación entre ambas manos.
        origen = np.mean([mano.puntos[0] for mano in presentes], axis=0)
        todos = np.concatenate([mano.puntos - origen for mano in presentes])
        escala = float(np.max(np.linalg.norm(todos[:, :2], axis=1)))
        escala = escala if escala > 0 else 1.0

        vectores = []
        for lado in ("left", "right"):
            mano = por_lado.get(lado)
            puntos = vacia if mano is None else (mano.puntos - origen) / escala
            vectores.append(puntos.flatten())
        return np.concatenate(vectores).astype(np.float32)

    @staticmethod
    def _puntos(landmarks: object) -> np.ndarray:
        return np.array(
            [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
        )

    @staticmethod
    def _normalizar(landmarks: object) -> np.ndarray:
        puntos = DetectorManos._puntos(landmarks)
        puntos -= puntos[0]  # La muñeca pasa a ser el origen.
        escala = float(np.max(np.linalg.norm(puntos[:, :2], axis=1)))
        if escala > 0:
            puntos /= escala
        return puntos.flatten()

    def cerrar(self) -> None:
        self._detector.close()

    def __enter__(self) -> "DetectorManos":
        return self

    def __exit__(self, *_: object) -> None:
        self.cerrar()
