import cv2
import numpy as np
import mediapipe as mp
import torch
import pyttsx3
from collections import deque
import os
import joblib
from threading import Thread
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(filename='sign_detection.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

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

def speak_async(text, engine):
    engine.say(text)
    engine.runAndWait()

def main():
    # Paths to model, scaler, and label encoder
    save_path = r"D:\Study\Project\SignLanguageRecognition\torch_data"
    model_path = os.path.join(save_path, 'sign_model.pt')
    scaler_path = os.path.join(save_path, 'scaler.pkl')

    # Check if files exist
    if not all(os.path.exists(p) for p in [model_path, scaler_path]):
        print(f"Error: One or more required files missing: {model_path}, {scaler_path}")
        return

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignClassifier(num_classes=29).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    # Manually recreate label encoder
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'delete', 'nothing']
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.1,  # Kept low for M, N detection
        min_tracking_confidence=0.5
    )

    # Initialize TTS
    tts_engine = initialize_tts()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize variables for sign detection and word formation
    sign_queue = deque(maxlen=30)  # Reduced to 30 frames (~1s at 30 FPS)
    word_buffer = []  # List of individual letters
    sentence = []     # List of words
    last_sign = None
    last_spoken = None
    sign_duration = 0
    last_action = None  # Track the last action (e.g., 'space')
    prob_threshold = 0.75
    frame_count = 0
    last_displayed_text = {"sign": "None", "text": "", "prob": 0.0}
    stability_threshold = 15  # Require 15 frames (~0.5s) for sign stability
    stable_sign_count = 0
    current_sign = None
    skip_frames = 1
    space_count = 0  # Track consecutive spaces
    last_space_frame = None  # Track the last frame with 'space'

    # Main loop with tqdm progress bar
    with tqdm(desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame_count += 1

            # Optional frame skip to reduce processing load
            if skip_frames > 1 and frame_count % skip_frames != 0:
                cv2.putText(frame, f"Sign: {last_displayed_text['sign']}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Text: {last_displayed_text['text']}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Prob: {last_displayed_text['prob']:.4f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Space: {space_count}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Visual feedback
                cv2.imshow('Sign Language Detection', frame)
                pbar.update(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Process frame for hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)

            sign_label = "None"
            max_prob = torch.tensor(0.0)
            raw_pred = -1
            hands_detected = bool(hand_results.multi_hand_landmarks)
            if hands_detected:
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
                    raw_pred = predicted.item()
                    if max_prob.item() > prob_threshold:
                        sign_label = label_encoder.inverse_transform([predicted.item()])[0]
                sign_queue.clear()  # Reset queue when hand is detected
            else:
                # Handle no-hand case (default to "None" unless confident it's "nothing")
                landmarks = np.zeros(63)
                landmarks = scaler.transform([landmarks])[0]
                x_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = model(x_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    max_prob, predicted = torch.max(probs, 1)
                    raw_pred = predicted.item()
                    predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
                    if max_prob.item() > prob_threshold and predicted_label == "nothing":
                        sign_label = "nothing"
                    sign_queue.clear()  # Reset queue when no hand to avoid carryover

            # Smooth predictions using a queue
            if sign_label != "None":
                sign_queue.append(sign_label)
                sign_label = max(set(sign_queue), key=sign_queue.count)

            # Stability check for sign acceptance
            if sign_label == current_sign and sign_label != "None":
                stable_sign_count += 1
            else:
                stable_sign_count = 0
                current_sign = sign_label

            # Word and sentence formation logic (only update if stable)
            if stable_sign_count >= stability_threshold and sign_label != last_sign:
                if sign_label == "space":
                    if word_buffer:
                        sentence.append("".join(word_buffer))
                        word_buffer = []
                    # Check for double space
                    if last_space_frame and (frame_count - last_space_frame <= 90):  # Extended to 90 frames (~3s)
                        space_count += 1
                    else:
                        space_count = 1
                    last_space_frame = frame_count

                    if space_count == 1:  # Single space
                        sentence.append(" ")  # Add space to separate words
                    elif space_count == 2:  # Double space
                        # Trigger TTS for the full sentence
                        speak_text = " ".join(sentence).strip()
                        if speak_text and speak_text != last_spoken:
                            Thread(target=speak_async, args=(speak_text, tts_engine)).start()
                            last_spoken = speak_text
                        sentence.append(".")  # Add sentence-ending full stop
                        # Reset for a new sentence
                        sentence = []
                        word_buffer = []
                        space_count = 0
                    last_action = "space"

                elif sign_label == "delete":
                    if word_buffer:
                        word_buffer.pop()  # Delete one letter from word_buffer
                    elif sentence and last_action == "space":  # Undo space by moving last word back as letters
                        if sentence and sentence[-1] == " ":
                            sentence.pop()  # Remove the space
                            if len(sentence) > 0:
                                last_word = sentence.pop()
                                word_buffer = list(last_word)  # Split into individual letters
                            last_action = None
                        elif sentence and sentence[-1] == ".":
                            sentence.pop()  # Remove the sentence-ending full stop
                    elif sentence:  # Delete one letter from the last word in sentence
                        last_word = sentence[-1]
                        if last_word == " " or last_word == ".":
                            sentence.pop()  # Remove space or full stop
                        elif len(last_word) > 1:
                            sentence[-1] = last_word[:-1]  # Remove last letter
                        else:
                            sentence.pop()  # Remove word if only one letter remains
                    last_action = "delete"
                elif sign_label == "nothing":
                    # Reset sentence on 'nothing'
                    sentence = []
                    word_buffer = []
                    space_count = 0
                    last_action = "nothing"
                else:
                    word_buffer.append(sign_label)
                    space_count = 0  # Reset space count on new letter
                    last_action = None
                last_sign = sign_label
                sign_duration = 1
            elif sign_label == last_sign:
                sign_duration += 1
                if sign_duration > 60:  # Increased to 60 frames (~2s at 30 FPS)
                    last_sign = None

            # Update current text
            current_text = "".join(sentence)  # Join with spaces and full stop

            # Update text buffer for display
            last_displayed_text = {
                "sign": sign_label,
                "text": current_text + "".join(word_buffer) if word_buffer else current_text,
                "prob": max_prob.item()
            }

            # Display on frame (always draw text to avoid blinking)
            cv2.putText(frame, f"Sign: {last_displayed_text['sign']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Text: {last_displayed_text['text']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Prob: {last_displayed_text['prob']:.4f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Space: {space_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Visual feedback

            # Show frame
            cv2.imshow('Sign Language Detection', frame)
            pbar.update(1)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()