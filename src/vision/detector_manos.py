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
    caracteristicas: np.ndarray
    lateralidad: str


class DetectorManos:
    """Pequeño envoltorio de MediaPipe Hands para cámara y entrenamiento."""

    def __init__(
        self,
        max_manos: int = 1,
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
        """Devuelve las manos detectadas y 63 valores relativos por mano."""
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
                    caracteristicas=self._normalizar(landmarks),
                    lateralidad=etiqueta,
                )
            )
        return manos

    @staticmethod
    def _normalizar(landmarks: object) -> np.ndarray:
        puntos = np.array(
            [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
        )
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

