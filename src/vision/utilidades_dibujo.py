"""Funciones de superposición para los fotogramas de cámara."""

import cv2
import mediapipe as mp
import numpy as np


def dibujar_mano(frame: np.ndarray, landmarks: object) -> None:
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
    )


def dibujar_etiqueta(
    frame: np.ndarray,
    texto: str,
    posicion: tuple[int, int] = (20, 45),
) -> None:
    cv2.putText(
        frame,
        texto,
        posicion,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )

