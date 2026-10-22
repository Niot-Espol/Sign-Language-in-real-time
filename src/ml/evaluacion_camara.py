import threading
import time
import unicodedata
from pathlib import Path

import cv2
import joblib
import numpy as np
import pygame

from src.vision.detector_manos import DetectorManos
from src.vision.utilidades_dibujo import dibujar_mano

CARPETA_AUDIOS = Path("assets/audios")
CARPETA_EVIDENCIAS = Path("docs/evidencias")
UMBRAL = 0.75
COOLDOWN = 2.5

audios = {
    "no": "assets/audios/no.mp3",
    "gracias": "assets/audios/gracias.mp3",
    "mama": "assets/audios/mama.mp3",
    "tomar": "assets/audios/tomar.mp3",
}


def limpiar(texto):
    base = unicodedata.normalize("NFKD", str(texto).strip().lower())
    return "".join(c for c in base if not unicodedata.combining(c)).replace(" ", "_")


def buscar_audio(clave):
    ruta = Path(audios.get(clave, ""))
    if ruta.is_file():
        return ruta
    if CARPETA_AUDIOS.exists():
        for archivo in sorted(CARPETA_AUDIOS.iterdir()):
            nombre = limpiar(archivo.stem)
            if nombre == clave or nombre.startswith(clave + "_"):
                return archivo
    return None


def sonar(ruta):
    try:
        pygame.mixer.Sound(str(ruta)).play()
    except Exception as error:
        print("Error con el audio:", error)


pygame.mixer.init()
modelo = joblib.load("models/modelo_senas.pkl")
CARPETA_EVIDENCIAS.mkdir(parents=True, exist_ok=True)

ultimo = ""
ultimo_tiempo = 0.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

print("Q para salir | G para guardar evidencia")

with DetectorManos(max_manos=2) as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        manos = detector.procesar(frame)
        for mano in manos:
            dibujar_mano(frame, mano.landmarks)

        etiqueta = None
        prob = 0.0

        if manos:
            vector = np.asarray(detector.vector_dos_manos(manos), dtype=float)
            n = getattr(modelo, "n_features_in_", len(vector))
            probs = modelo.predict_proba(vector[:n].reshape(1, -1))[0]
            i = int(np.argmax(probs))
            prob = float(probs[i])
            if prob >= UMBRAL:
                etiqueta = limpiar(modelo.classes_[i])

        ahora = time.time()

        if etiqueta:
            if etiqueta != ultimo or ahora - ultimo_tiempo > COOLDOWN:
                ruta = buscar_audio(etiqueta)
                if ruta:
                    threading.Thread(target=sonar, args=(ruta,), daemon=True).start()
                else:
                    print("Sin audio para:", etiqueta)
                ultimo = etiqueta
                ultimo_tiempo = ahora

            cv2.rectangle(frame, (20, 20), (520, 95), (0, 160, 0), -1)
            texto = f"{etiqueta.replace('_', ' ').upper()} ({prob * 100:.1f}%)"
            cv2.putText(frame, texto, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (20, 20), (330, 75), (50, 50, 50), -1)
            cv2.putText(frame, "Esperando sena...", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Evaluacion en vivo", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break
        if tecla == ord("g") and etiqueta:
            marca = time.strftime("%H%M%S")
            cv2.imwrite(str(CARPETA_EVIDENCIAS / f"{etiqueta}_luis_{marca}.jpg"), frame)
            print("Evidencia guardada:", etiqueta)

cap.release()
cv2.destroyAllWindows()