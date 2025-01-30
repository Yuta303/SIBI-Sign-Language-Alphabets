import cv2
import mediapipe as mp
import joblib
import numpy as np

# Memuat model yang sudah disimpan
model1 = joblib.load('hand_tracking_training.pkl')  # Model 1

# Inisialisasi MediaPipe Hands dan Drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Membuka kamera dengan resolusi tinggi
cap = cv2.VideoCapture(0)
# Inisialisasi MediaPipe Hands dan Drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Menyeting jendela OpenCV menjadi full screen
cv2.namedWindow("Hand Tracking with Gesture Recognition", cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Membalikkan frame secara horizontal untuk tampilan mirror
    frame = cv2.flip(frame, 1)

    # Mengonversi frame ke RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Menggambar landmark tangan dan memproses data
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Ekstraksi fitur koordinat x, y, z dari landmark
            features = []
            for landmark in hand_landmarks.landmark:
                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z)

            # Mengonversi fitur ke format yang dapat diterima model
            features = np.array(features).reshape(1, -1)  # 1 sample, 63 features

            # Prediksi gesture menggunakan model1
            prediction = model1.predict(features)[0]  # Model 1

            # Menampilkan hasil prediksi di layar rekaman
            gesture_text = f"HURUF : {prediction}"

            # Menampilkan hasil prediksi di layar
            cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Menampilkan frame
    cv2.imshow('Hand Tracking with Gesture Recognition', frame)

    # Menunggu keypress 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup video capture dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()