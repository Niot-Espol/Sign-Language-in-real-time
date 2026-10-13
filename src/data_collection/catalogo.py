"""Catálogo compartido y validado de señas."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[2]
RUTA_CATALOGO = RAIZ / "config" / "catalogo.json"


@dataclass(frozen=True)
class Sena:
    etiqueta: str
    slug: str
    tipo: str
    manos_requeridas: int | None


def cargar_catalogo(ruta: Path = RUTA_CATALOGO) -> list[Sena]:
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    senas = [Sena(**item) for item in contenido["palabras"]]
    if len(senas) != 20:
        raise ValueError("El catálogo debe contener exactamente 20 señas.")
    if len({s.etiqueta for s in senas}) != 20 or len({s.slug for s in senas}) != 20:
        raise ValueError("Las etiquetas y los slugs del catálogo deben ser únicos.")
    for sena in senas:
        if sena.tipo not in {"estatica", "dinamica", "por_definir"}:
            raise ValueError(f"Tipo inválido para {sena.etiqueta}: {sena.tipo}")
        if sena.manos_requeridas not in {None, 1, 2}:
            raise ValueError(f"Cantidad de manos inválida para {sena.etiqueta}.")
    return senas


def buscar_sena(texto: str, catalogo: list[Sena]) -> Sena:
    buscado = texto.strip().casefold()
    for sena in catalogo:
        if buscado in {sena.etiqueta.casefold(), sena.slug.casefold()}:
            return sena
    raise ValueError(f"Seña fuera del catálogo oficial: {texto}")

