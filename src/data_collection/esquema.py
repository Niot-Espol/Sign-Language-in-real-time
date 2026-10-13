"""Definición única del esquema tabular para todo el equipo."""

VERSION_ESQUEMA = 2


def columnas_landmarks() -> list[str]:
    return [
        f"{lado}_{eje}{indice}"
        for lado in ("left", "right")
        for indice in range(21)
        for eje in ("x", "y", "z")
    ]


def columnas_dataset() -> list[str]:
    return [*columnas_landmarks(), "label"]

