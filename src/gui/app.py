"""Ventana principal del traductor."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from src.gui.componentes import PanelResultado
from src.ml.clasificador import ClasificadorSenas
from src.vision.detector_manos import DetectorManos
from src.vision.utilidades_dibujo import dibujar_mano


class AplicacionTraductor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Traductor de lenguaje de señas")
        self.geometry("900x700")
        self.configure(bg="#0f172a")
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.video = tk.Label(self, bg="#0f172a")
        self.video.pack(fill="both", expand=True, padx=20, pady=20)
        self.resultado = PanelResultado(self)
        self.resultado.pack(fill="x", padx=20, pady=(0, 20))

        self.detector = DetectorManos()
        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            self.detector.cerrar()
            raise RuntimeError("No se pudo abrir la cámara.")

        raiz = Path(__file__).resolve().parents[2]
        try:
            self.clasificador = ClasificadorSenas(
                raiz / "models/modelo_senas.pkl", raiz / "models/labels.json"
            )
        except FileNotFoundError:
            self.clasificador = None
            self.resultado.actualizar("Modo cámara: falta entrenar modelo_senas.pkl")

        self.after(10, self._actualizar_frame)

    def _actualizar_frame(self) -> None:
        ok, frame = self.captura.read()
        if not ok:
            self.resultado.actualizar("No se pudo leer la cámara")
            self.after(100, self._actualizar_frame)
            return

        frame = cv2.flip(frame, 1)
        manos = self.detector.procesar(frame)
        if manos:
            mano = manos[0]
            dibujar_mano(frame, mano.landmarks)
            if self.clasificador is not None:
                etiqueta, confianza = self.clasificador.predecir(mano.caracteristicas)
                self.resultado.actualizar(etiqueta, confianza)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(rgb)
        imagen.thumbnail((860, 560))
        foto = ImageTk.PhotoImage(imagen)
        self.video.configure(image=foto)
        self.video.image = foto
        self.after(15, self._actualizar_frame)

    def cerrar(self) -> None:
        if self.captura.isOpened():
            self.captura.release()
        self.detector.cerrar()
        self.destroy()


def ejecutar() -> None:
    AplicacionTraductor().mainloop()

