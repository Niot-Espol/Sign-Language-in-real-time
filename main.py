import os
import pickle
import cv2
import mediapipe as mp
import numpy as np


DIR_PROYECTO = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(DIR_PROYECTO, 'models', 'modelo_señas.pkl')


with open(RUTA_MODELO, 'rb') as f:
    modelo = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            fila = []
            for landmark in hand_landmarks.landmark:
                fila.extend([landmark.x, landmark.y, landmark.z])

            
            prediccion = modelo.predict([fila])[0]

           
            cv2.putText(frame, f"Seña: {prediccion}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Traductor de Lenguaje de Señas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()