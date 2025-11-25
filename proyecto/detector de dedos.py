# ProyectoA - Detector de manos y dedos levantados en WebCam

import cv2
import mediapipe as medP

# Instanciamos mediapipe
mpManos = medP.solutions.hands
manos = mpManos.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDibujo = medP.solutions.drawing_utils

# Lista para almacenar el dibujo
puntos_dibujo = []

# Método para contar los dedos levantados en la imagen
def contarDedos(puntosReferencia, mano):
    puntaDedo = [
        mpManos.HandLandmark.THUMB_TIP,
        mpManos.HandLandmark.INDEX_FINGER_TIP,
        mpManos.HandLandmark.MIDDLE_FINGER_TIP,
        mpManos.HandLandmark.RING_FINGER_TIP,
        mpManos.HandLandmark.PINKY_TIP
    ]

    puntoMedioDedo = [
        mpManos.HandLandmark.THUMB_IP,
        mpManos.HandLandmark.INDEX_FINGER_PIP,
        mpManos.HandLandmark.MIDDLE_FINGER_PIP,
        mpManos.HandLandmark.RING_FINGER_PIP,
        mpManos.HandLandmark.PINKY_PIP
    ]

    dedosLevantados = 0

    # Mano derecha
    if mano == 0:
        if puntosReferencia.landmark[puntaDedo[0]].x < puntosReferencia.landmark[puntoMedioDedo[0]].x:
            dedosLevantados += 1
    else:
        # Mano izquierda
        if puntosReferencia.landmark[puntaDedo[0]].x > puntosReferencia.landmark[puntoMedioDedo[0]].x:
            dedosLevantados += 1

    # Otros dedos
    for i in range(1, 5):
        if puntosReferencia.landmark[puntaDedo[i]].y < puntosReferencia.landmark[puntoMedioDedo[i]].y:
            dedosLevantados += 1

    return dedosLevantados


# Abrimos ventana con captura de vídeo de webcam
capturaVideo = cv2.VideoCapture(0)

while capturaVideo.isOpened():
    ok, imagen = capturaVideo.read()
    if not ok:
        continue

    # Procesamiento de imagen
    imagenRGB = cv2.cvtColor(cv2.flip(imagen, 1), cv2.COLOR_BGR2RGB)
    resultado = manos.process(imagenRGB)
    imagen = cv2.cvtColor(imagenRGB, cv2.COLOR_RGB2BGR)

    if resultado.multi_hand_landmarks:
        for idx, puntosReferencia in enumerate(resultado.multi_hand_landmarks):
            mano = idx % 2

            # Dibujar landmarks
            mpDibujo.draw_landmarks(imagen, puntosReferencia, mpManos.HAND_CONNECTIONS)

            # Contar dedos
            dedos = contarDedos(puntosReferencia, mano)

            # Coords del índice
            h, w, c = imagen.shape
            indice_tip = puntosReferencia.landmark[mpManos.HandLandmark.INDEX_FINGER_TIP]
            x = int(indice_tip.x * w)
            y = int(indice_tip.y * h)

            # 🎨 Si solo está levantado el índice → modo dibujo
            if dedos == 1:
                puntos_dibujo.append((x, y))

            # ✋ Si no hay dedos levantados → borrar dibujo
            elif dedos == 0:
                puntos_dibujo = []

            # Mostrar número de dedos
            cv2.putText(imagen, f"Dedos: {dedos}", (10, 50 + (30*idx)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Dibujar trazos acumulados
    for i in range(1, len(puntos_dibujo)):
        cv2.line(imagen, puntos_dibujo[i-1], puntos_dibujo[i], (0, 0, 255), 6)

    cv2.imshow("Dibujar con la Mano", imagen)

    if cv2.waitKey(5) & 0xFF == ord('s'):
        break

capturaVideo.release()
cv2.destroyAllWindows()
