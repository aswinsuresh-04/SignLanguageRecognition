import cv2
import numpy as np
import mediapipe as mp
import torch
import pyttsx3
from deepface import DeepFace
from collections import deque
import os
import joblib
from threading import Thread

class SignClassifier(torch.nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(63, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.8)
    return engine

def map_emotion_to_tts_properties(emotion):
    return {
        'happy': {'rate': 180, 'volume': 0.9},
        'sad': {'rate': 120, 'volume': 0.6},
        'angry': {'rate': 200, 'volume': 1.0},
        'neutral': {'rate': 150, 'volume': 0.8},
        'surprise': {'rate': 190, 'volume': 0.9},
        'fear': {'rate': 170, 'volume': 0.7},
        'disgust': {'rate': 160, 'volume': 0.8}
    }.get(emotion.lower(), {'rate': 150, 'volume': 0.8})

def speak_async(text, engine, props):
    engine.setProperty('rate', props['rate'])
    engine.setProperty('volume', props['volume'])
    engine.say(text)
    engine.runAndWait()

def main():
    save_path = r"D:\Study\Project\SignLAnguage\torch_data"
    model_path = os.path.join(save_path, 'sign_model.pt')
    label_encoder_path = os.path.join(save_path, 'label_encoder.pt')
    scaler_path = os.path.join(save_path, 'scaler.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignClassifier(num_classes=29).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    label_encoder = torch.load(label_encoder_path, weights_only=False)
    scaler = joblib.load(scaler_path)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5
    )
    tts_engine = initialize_tts()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sign_queue = deque(maxlen=10)
    emotion_queue = deque(maxlen=10)
    last_spoken = None
    emotion_frame_skip = 5
    frame_count = 0
    word_buffer = []
    sentence = []
    last_sign = None
    sign_duration = 0
    current_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)

        sign_label = "None"
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = scaler.transform([landmarks])[0]
            x_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(x_tensor)
                probs = torch.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probs, 1)
                if max_prob.item() > 0.7:
                    sign_label = label_encoder.inverse_transform([predicted.item()])[0]
            sign_queue.append(sign_label)
            sign_label = max(set(sign_queue), key=sign_queue.count)
        else:
            sign_queue.append("None")
            sign_label = max(set(sign_queue), key=sign_queue.count)

        # Word/sentence formation
        if sign_label != "None" and sign_label != last_sign:
            if sign_label == "space":
                if word_buffer:
                    sentence.append("".join(word_buffer))
                    word_buffer = []
            elif sign_label == "delete":
                if word_buffer:
                    word_buffer.pop()
            elif sign_label == "nothing":
                pass
            else:
                word_buffer.append(sign_label)
            last_sign = sign_label
            sign_duration = 1
        elif sign_label == last_sign:
            sign_duration += 1
            if sign_duration > 10:  # Adjustable threshold
                last_sign = None

        current_text = " ".join(sentence + ["".join(word_buffer)])
        cv2.putText(frame, f"Text: {current_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Emotion detection
        if frame_count % emotion_frame_skip == 0:
            try:
                emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']
            except:
                emotion = 'neutral'
            emotion_queue.append(emotion)
        emotion = max(set(emotion_queue), key=emotion_queue.count)
        frame_count += 1

        # Display
        cv2.putText(frame, f"Sign: {sign_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak individual signs (temporary, for debugging)
        if sign_label != "None" and sign_label != last_spoken:
            tts_props = map_emotion_to_tts_properties(emotion)
            Thread(target=speak_async, args=(sign_label, tts_engine, tts_props)).start()
            last_spoken = sign_label

        # Speak when a word is complete
        if sentence and not word_buffer and current_text != last_spoken:
            tts_props = map_emotion_to_tts_properties(emotion)
            Thread(target=speak_async, args=(current_text, tts_engine, tts_props)).start()
            last_spoken = current_text

        cv2.imshow('Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()