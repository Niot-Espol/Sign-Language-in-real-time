"""Valida y consolida los CSV individuales sin mezclar formatos incompatibles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_collection.catalogo import cargar_catalogo
from src.data_collection.esquema import columnas_dataset, columnas_landmarks


def _identidad_archivo(ruta: Path) -> tuple[str, str]:
    nombre = ruta.stem
    for sena in sorted(cargar_catalogo(), key=lambda s: len(s.slug), reverse=True):
        prefijo = f"{sena.slug}_"
        if nombre.startswith(prefijo) and nombre[len(prefijo):]:
            return sena.etiqueta, nombre[len(prefijo):]
    raise ValueError(f"Nombre inválido: {ruta.name}; use <palabra>_<integrante>.csv")


def consolidar(entrada: Path, salida: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    archivos = sorted(entrada.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay CSV en {entrada}")

    partes: list[pd.DataFrame] = []
    metadatos: list[pd.DataFrame] = []
    errores: list[str] = []
    esperado = columnas_dataset()
    features = columnas_landmarks()

    for ruta in archivos:
        try:
            etiqueta, integrante = _identidad_archivo(ruta)
            datos = pd.read_csv(ruta)
            if datos.columns.tolist() != esperado:
                raise ValueError(
                    f"esquema inválido: {len(datos.columns) - 1} características; se esperan 126"
                )
            if datos.empty:
                raise ValueError("archivo vacío")
            if set(datos["label"].astype(str)) != {etiqueta}:
                raise ValueError(f"la etiqueta interna no coincide con '{etiqueta}'")
            numericos = datos[features].apply(pd.to_numeric, errors="coerce")
            validas = np.isfinite(numericos.to_numpy()).all(axis=1)
            datos = datos.loc[validas].copy()
            datos[features] = numericos.loc[validas]
            datos = datos.drop_duplicates(subset=features).reset_index(drop=True)
            if datos.empty:
                raise ValueError("no quedaron filas válidas después de limpiar")
            inicio = sum(len(parte) for parte in partes)
            partes.append(datos)
            metadatos.append(pd.DataFrame({
                "sample_id": range(inicio, inicio + len(datos)),
                "integrante": integrante,
                "archivo": ruta.name,
            }))
        except Exception as exc:
            errores.append(f"- {ruta.name}: {exc}")

    if errores:
        raise ValueError("No se consolidó por estos errores:\n" + "\n".join(errores))

    dataset = pd.concat(partes, ignore_index=True)
    metadata = pd.concat(metadatos, ignore_index=True)
    salida.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(salida, index=False)
    metadata.to_csv(salida.with_name("metadata_consolidado.csv"), index=False)
    return dataset, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--salida", type=Path,
        default=Path("data/processed/dataset_consolidado.csv"),
    )
    args = parser.parse_args()
    dataset, metadata = consolidar(args.entrada, args.salida)
    print(
        f"OK: {len(dataset)} filas, {dataset['label'].nunique()} palabras, "
        f"{metadata['integrante'].nunique()} integrantes."
    )


if __name__ == "__main__":
    main()

