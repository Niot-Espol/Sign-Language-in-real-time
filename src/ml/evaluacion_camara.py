import time
import threading
import cv2
import joblib
import numpy as np
import pygame

from src.vision.detector_manos import DetectorManos
from src.vision.utilidades_dibujo import dibujar_mano

pygame.mixer.init()
modelo = joblib.load("models/modelo_senas.pkl")

audios = {
    "no": "assets/audios/no.mp3",
    "gracias": "assets/audios/gracias.mp3",
    "mama": "assets/audios/mama.mp3",
    "tomar/beber": "assets/audios/tomar_beber.mp3",
}

ultimo = ""
ultimo_tiempo = 0


def quitar_tildes(texto):
    texto = texto.lower()
    for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("_", "/")]:
        texto = texto.replace(a, b)
    return texto


def sonar(ruta):
    try:
        pygame.mixer.Sound(ruta).play()
    except Exception as e:
        print("Error con el audio:", e)


cap = cv2.VideoCapture(0)

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
        prob = 0

        if len(manos) > 0:
            vector = np.array(detector.vector_dos_manos(manos))
            n = getattr(modelo, "n_features_in_", len(vector))
            probs = modelo.predict_proba(vector[:n].reshape(1, -1))[0]
            i = np.argmax(probs)
            prob = probs[i]
            if prob >= 0.75:
                etiqueta = quitar_tildes(str(modelo.classes_[i]))

        ahora = time.time()

        if etiqueta:
            if etiqueta != ultimo or ahora - ultimo_tiempo > 2.5:
                if etiqueta in audios:
                    threading.Thread(target=sonar, args=(audios[etiqueta],), daemon=True).start()
                ultimo = etiqueta
                ultimo_tiempo = ahora

            cv2.rectangle(frame, (20, 20), (520, 95), (0, 160, 0), -1)
            texto = f"{etiqueta.upper()} ({prob * 100:.1f}%)"
            cv2.putText(frame, texto, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (20, 20), (330, 75), (50, 50, 50), -1)
            cv2.putText(frame, "Esperando sena...", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Evaluacion en vivo", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break
        if tecla == ord("g") and etiqueta:
            nombre = etiqueta.replace("/", "_")
            cv2.imwrite(f"docs/evidencias/{nombre}_luis.jpg", frame)

cap.release()
cv2.destroyAllWindows()