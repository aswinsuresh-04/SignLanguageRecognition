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
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import time
import re

# Set up logging
logging.basicConfig(filename='sign_detection.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Sign detection model (unchanged)
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

def speak_async(text, engine, speaking_flag):
    speaking_flag[0] = True  # Mark as speaking
    engine.say(text)
    engine.runAndWait()
    speaking_flag[0] = False  # Mark as done speaking

def emotion_classification(face_roi):
    """Guess the emotion from a face part of the video."""
    face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    inputs = processor(images=face_pil, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
        max_prob, pred_idx = torch.max(probs, dim=0)
        emotion_label = emotion_labels[str(pred_idx.item())]
        if emotion_label == "Tease" and max_prob < 0.7:  # If it’s unsure about Tease, call it Surprise
            emotion_label = "Surprise"
        return emotion_label, max_prob.item()  # Return float

def get_most_common_emotion(emotions):
    """Return the most common emotion from the deque, defaulting to 'None' if empty."""
    if not emotions:
        return "None"
    return max(set(emotions), key=emotions.count)

def main():
    # Where to find the sign model and scaler
    save_path = r"D:\Study\Project\SignLanguageRecognition\torch_data"
    sign_model_path = os.path.join(save_path, 'sign_model.pt')
    scaler_path = os.path.join(save_path, 'scaler.pkl')

    # Check if files exist
    if not all(os.path.exists(p) for p in [sign_model_path, scaler_path]):
        print(f"Error: Missing files: {sign_model_path} or {scaler_path}")
        return

    # Use GPU if available
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load sign model
    sign_model = SignClassifier(num_classes=29).to(device)
    sign_model.load_state_dict(torch.load(sign_model_path, weights_only=True, map_location=device))
    sign_model.eval()

    # Load the emotion model (SigLIP2)
    model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
    global model, processor, emotion_labels
    model = SiglipForImageClassification.from_pretrained(model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()

    # Set up sign labels
    sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
                   'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'delete', 'nothing']
    label_encoder = LabelEncoder()
    label_encoder.fit(sign_labels)

    # Emotion labels from the model
    emotion_labels = {
        "0": "Tease", "1": "Angry", "2": "Happy", "3": "Neutral",
        "4": "Sad", "5": "Surprise"
    }

    # Load scaler for sign detection
    scaler = joblib.load(scaler_path)

    # Set up hand and face detection
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.5
    )
    face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=0.5
    )

    # Set up text-to-speech
    tts_engine = initialize_tts()

    # Start the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Can’t open webcam.")
        return

    # Set up variables for signs and emotions
    sign_queue = deque(maxlen=30)
    word_buffer = []
    sentence = []
    last_sign = None
    last_spoken = None
    sign_duration = 0
    last_action = None
    prob_threshold = 0.75
    frame_count = 0
    last_displayed_text = {"sign": "None", "text": "", "prob": 0.0}
    stability_threshold = 15
    stable_sign_count = 0
    current_sign = None
    skip_frames = 1
    space_count = 0
    last_space_frame = None
    emotion_history = deque(maxlen=15)  # Reduced from 30 to 15 for faster emotion switching
    caption_text = ""  # For displaying the final sentence with emotion
    caption_start_time = None  # To track when the caption was displayed
    status_message = "Waiting for Signs"  # Status message for top-right corner
    speaking_flag = [False]  # Flag to track if TTS is speaking

    # Main loop to process video
    with tqdm(desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can’t capture frame.")
                break

            frame_count += 1

            # Skip some frames to make it faster
            if skip_frames > 1 and frame_count % skip_frames != 0:
                # Update status message (same as previous frame since we're skipping)
                if not speaking_flag[0]:  # Only update status if TTS is not speaking
                    status_message = "Waiting for Signs"
                cv2.putText(frame, f"Sign: {last_displayed_text['sign']}", (10, 30), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
                cv2.putText(frame, f"Text: {last_displayed_text['text']}", (10, 60), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
                cv2.putText(frame, f"Prob: {last_displayed_text['prob']:.4f}", (10, 90), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
                cv2.putText(frame, f"Space: {space_count}", (10, 120), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
                cv2.putText(frame, f"Emotion: {get_most_common_emotion(emotion_history)}", (10, 150), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)  # Changed to green
                # Display the status message in the top-right corner
                text_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)[0]
                text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from right edge
                cv2.putText(frame, status_message, (text_x, 30), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)
                # Display the caption at the bottom if it exists
                current_time = time.time()
                if caption_text and caption_start_time and (current_time - caption_start_time < 5):
                    cv2.putText(frame, caption_text, (10, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
                else:
                    caption_text = ""  # Clear the caption after 5 seconds
                    caption_start_time = None
                cv2.imshow('Sign Language Detection', frame)
                pbar.update(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Detect hands and faces
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            face_results = face_detection.process(frame_rgb)

            sign_label = "None"
            max_prob = 0.0  # Initialize as float
            raw_pred = -1
            hands_detected = bool(hand_results.multi_hand_landmarks)
            face_detected = bool(face_results.detections)

            # Update status message based on sign detection, but only if TTS is not speaking
            if not speaking_flag[0]:
                if hands_detected or sign_label != "None":
                    status_message = "Showing Signs"

            if hands_detected:
                hand = hand_results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = scaler.transform([landmarks])[0]
                x_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = sign_model(x_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    max_prob_tensor, predicted = torch.max(probs, 1)
                    raw_pred = predicted.item()
                    if max_prob_tensor.item() > prob_threshold:
                        sign_label = label_encoder.inverse_transform([predicted.item()])[0]
                    max_prob = max_prob_tensor.item()  # Convert to float
                sign_queue.clear()

            # Detect emotion from face
            if face_detected:
                face = face_results.detections[0]
                bbox = face.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                crop_size = min(w, h)
                center_x, center_y = x + w // 2, y + h // 2
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(iw, center_x + crop_size // 2)
                y2 = min(ih, center_y + crop_size // 2)
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0 and crop_size > 0:  # Ensure valid face region
                    emotion_label, max_prob = emotion_classification(face_roi)
                    emotion_history.append(emotion_label)  # Store the detected emotion
            else:
                emotion_history.append("None")  # Append 'None' if no face detected

            if not hands_detected:
                landmarks = np.zeros(63)
                landmarks = scaler.transform([landmarks])[0]
                x_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = sign_model(x_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    max_prob_tensor, predicted = torch.max(probs, 1)
                    raw_pred = predicted.item()
                    if max_prob_tensor.item() > prob_threshold and label_encoder.inverse_transform([predicted.item()])[0] == "nothing":
                        sign_label = "nothing"
                    max_prob = max_prob_tensor.item()  # Convert to float
                sign_queue.clear()
                if not speaking_flag[0]:  # Only reset status if TTS is not speaking
                    status_message = "Waiting for Signs"  # No hands detected, waiting for signs

            # Smooth out sign predictions
            if sign_label != "None":
                sign_queue.append(sign_label)
                sign_label = max(set(sign_queue), key=sign_queue.count)

            # Check if the sign is stable
            if sign_label == current_sign and sign_label != "None":
                stable_sign_count += 1
            else:
                stable_sign_count = 0
                current_sign = sign_label

            # Build words and sentences from signs
            if stable_sign_count >= stability_threshold and sign_label != last_sign:
                if sign_label == "space":
                    if word_buffer:
                        sentence.append("".join(word_buffer))
                        word_buffer = []
                    if last_space_frame and (frame_count - last_space_frame <= 90):
                        space_count += 1
                    else:
                        space_count = 1
                    last_space_frame = frame_count

                    if space_count == 1:
                        sentence.append(" ")
                    elif space_count == 2:
                        speak_text = " ".join(sentence).strip()
                        # Clean up multiple spaces in speak_text
                        speak_text = re.sub(r'\s+', ' ', speak_text).strip()
                        if speak_text and speak_text != last_spoken:
                            # Determine the most common emotion from the history
                            dominant_emotion = get_most_common_emotion(emotion_history)
                            emotion_phrase = f"with a {dominant_emotion.lower()} face" if dominant_emotion != "None" else "with no emotion"
                            full_speech = f"{speak_text}, {emotion_phrase}"
                            # Start TTS and set status to "Conveying the Message"
                            Thread(target=speak_async, args=(full_speech, tts_engine, speaking_flag)).start()
                            status_message = "Conveying the Message"  # Set status when TTS starts
                            last_spoken = speak_text
                            # Set the caption text with the dominant emotion
                            caption_emotion = dominant_emotion.upper() if dominant_emotion != "None" else "NO EMOTION"
                            caption_text = f"{speak_text} ({caption_emotion})"
                            caption_start_time = time.time()
                            # Clear the emotion history after using it
                            emotion_history.clear()
                        sentence.append(".")
                        sentence = []
                        word_buffer = []
                        space_count = 0
                    last_action = "space"

                elif sign_label == "delete":
                    if word_buffer:
                        word_buffer.pop()
                    elif sentence and last_action == "space":
                        if sentence and sentence[-1] == " ":
                            sentence.pop()
                            if len(sentence) > 0:
                                last_word = sentence.pop()
                                word_buffer = list(last_word)
                            last_action = None
                        elif sentence and sentence[-1] == ".":
                            sentence.pop()
                    elif sentence:
                        last_word = sentence[-1]
                        if last_word == " " or last_word == ".":
                            sentence.pop()
                        elif len(last_word) > 1:
                            sentence[-1] = last_word[:-1]
                        else:
                            sentence.pop()
                    last_action = "delete"
                elif sign_label == "nothing":
                    sentence = []
                    word_buffer = []
                    space_count = 0
                    last_action = "nothing"
                    if not speaking_flag[0]:  # Only reset status if TTS is done
                        status_message = "Waiting for Signs"  # Reset status when sentence is cleared
                else:
                    word_buffer.append(sign_label)
                    space_count = 0
                    last_action = None
                last_sign = sign_label
                sign_duration = 1
            elif sign_label == last_sign:
                sign_duration += 1
                if sign_duration > 60:
                    last_sign = None

            current_text = "".join(sentence)

            last_displayed_text = {
                "sign": sign_label,
                "text": current_text + "".join(word_buffer) if word_buffer else current_text,
                "prob": max_prob  # Already a float
            }

            # Show everything on the screen
            cv2.putText(frame, f"Sign: {last_displayed_text['sign']}", (10, 30), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
            cv2.putText(frame, f"Text: {last_displayed_text['text']}", (10, 60), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
            cv2.putText(frame, f"Prob: {last_displayed_text['prob']:.4f}", (10, 90), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
            cv2.putText(frame, f"Space: {space_count}", (10, 120), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)  # Changed to red
            cv2.putText(frame, f"Emotion: {get_most_common_emotion(emotion_history)}", (10, 150), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)  # Changed to green

            # Display the status message in the top-right corner
            text_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from right edge
            cv2.putText(frame, status_message, (text_x, 30), 
                       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)

            # Display the caption at the bottom if it exists
            current_time = time.time()
            if caption_text and caption_start_time and (current_time - caption_start_time < 5):
                cv2.putText(frame, caption_text, (10, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
            else:
                caption_text = ""  # Clear the caption after 5 seconds
                caption_start_time = None

            cv2.imshow('Sign Language Detection', frame)
            pbar.update(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_detection.close()

if __name__ == "__main__":
    main()
