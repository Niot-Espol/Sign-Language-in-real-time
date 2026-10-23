import cv2
import csv
import os
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

nombre_seña = "tu"
ruta_csv = f"docs/evidencias/{nombre_seña}.csv"

os.makedirs("docs/evidencias", exist_ok=True)

cap = cv2.VideoCapture(1)

print(f"[INFO] Grabando datos para la seña: {nombre_seña}")
print("[INSTRUCCIÓN] Abre tu foto en 'docs/referencias/' como guía, ponte en postura, y presiona 's' para guardar fotogramas. Presiona 'q' para salir.")

with open(ruta_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    
    header = []
    for i in range(21):
        header += [f'x_{i}', f'y_{i}', f'z_{i}']
    writer.writerow(header)

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
                # Dibujar los 21 puntos clave en la pantalla
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                
                fila_coordenadas = []
                for landmark in hand_landmarks.landmark:
                    fila_coordenadas.extend([landmark.x, landmark.y, landmark.z])

                
                cv2.putText(frame, f"Seña: {nombre_seña} (Presiona 's')", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    writer.writerow(fila_coordenadas)
                    print(f"[REGISTRADO] Muestra guardada para '{nombre_seña}'")

      
        cv2.imshow('Recolección de Señas - MediaPipe', frame)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()