from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_collection.catalogo import buscar_sena, cargar_catalogo
from src.data_collection.consolidar_dataset import consolidar
from src.data_collection.esquema import columnas_dataset, columnas_landmarks
from src.vision.detector_manos import DetectorManos, ResultadoMano


class BaseEquipoTests(unittest.TestCase):
    def test_catalogo_oficial(self) -> None:
        catalogo = cargar_catalogo()
        self.assertEqual(len(catalogo), 20)
        self.assertEqual(buscar_sena("papa", catalogo).etiqueta, "papá")
        self.assertEqual(buscar_sena("tu", catalogo).etiqueta, "tu")

    def test_vector_fijo_con_relacion_entre_manos(self) -> None:
        puntos_izq = np.zeros((21, 3), dtype=np.float32)
        puntos_der = np.zeros((21, 3), dtype=np.float32)
        puntos_izq[:, 0] = 0.25
        puntos_der[:, 0] = 0.75
        izq = ResultadoMano(None, puntos_izq, puntos_izq.flatten(), "Left")
        der = ResultadoMano(None, puntos_der, puntos_der.flatten(), "Right")
        vector = DetectorManos.vector_dos_manos([der, izq])
        self.assertEqual(vector.shape, (126,))
        self.assertLess(vector[0], 0)
        self.assertGreater(vector[63], 0)

    def test_consolidacion_y_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            entrada = raiz / "raw"
            entrada.mkdir()
            fila = {columna: float(i) for i, columna in enumerate(columnas_landmarks())}
            fila["label"] = "hola"
            pd.DataFrame([fila], columns=columnas_dataset()).to_csv(
                entrada / "hola_ana.csv", index=False
            )
            dataset, metadata = consolidar(entrada, raiz / "dataset.csv")
            self.assertEqual(dataset.shape, (1, 127))
            self.assertEqual(metadata.loc[0, "integrante"], "ana")


if __name__ == "__main__":
    unittest.main()
