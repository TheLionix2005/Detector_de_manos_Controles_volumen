# volumen.py – Control de volumen + Play/Pause con Mediapipe y OpenCV

import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import keyboard

# ===============================
# FUNCION PLAY/PAUSE
# ===============================
def play_pause():
    keyboard.send("play/pause media")

# ===============================
# FUNCION PARA OBTENER LA APP QUE SUENA
# ===============================
def get_active_audio_app():
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        try:
            if session.State == 1:  # 1 = reproduciendo audio
                return session._ctl.QueryInterface(ISimpleAudioVolume)
        except:
            continue
    return None

# ===============================
# CONFIG Mediapipe
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

# ===============================
# DETECTAR DEDOS ARRIBA
# ===============================
def fingers_up(hand, handedness):
    fingers = []

    # Pulgar
    thumb_tip = hand.landmark[4].x
    thumb_ip = hand.landmark[3].x

    if handedness == "Right":
        fingers.append(1 if thumb_tip < thumb_ip else 0)
    else:
        fingers.append(1 if thumb_tip > thumb_ip else 0)

    # Otros dedos
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if hand.landmark[tip].y < hand.landmark[tip - 2].y else 0)

    return fingers

paused = False

# ===============================
# LOOP PRINCIPAL
# ===============================
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness_data in zip(results.multi_hand_landmarks,
                                                   results.multi_handedness):

            handedness = handedness_data.classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ------------------------------
            # Detectar dedos
            # ------------------------------
            dedos = fingers_up(hand_landmarks, handedness)
            total_dedos = sum(dedos)

            # Puño → pausa
            if total_dedos == 0 and not paused:
                play_pause()
                paused = True

            # Mano abierta → play
            if total_dedos == 5 and paused:
                play_pause()
                paused = False

            # ------------------------------
            # Control REAL del volumen
            # ------------------------------
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            y_tip = int(index_tip.y * h)

            volumen = 1 - (y_tip / h)
            volumen = max(0, min(volumen, 1))

            # Volumen de la APP que está sonando
            active_volume = get_active_audio_app()
            if active_volume:
                active_volume.SetMasterVolume(volumen, None)

            cv2.putText(frame, f"Volumen App: {int(volumen * 100)}%",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

    cv2.imshow("Control de Volumen con la Mano", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cam.release()
cv2.destroyAllWindows()
