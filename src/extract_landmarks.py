import os
import cv2
import numpy as np
import mediapipe as mp
import torch
from tqdm import tqdm
from collections import Counter
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

logging.basicConfig(filename='landmark_extraction.log', level=logging.INFO)

def augment_image(image):
    angle = np.random.uniform(-10, 10)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    return image

def extract_landmarks():
    # Paths
    data_dir = r"D:\Study\Project\SignLAnguage\asl_alphabet_train\asl_alphabet_train"
    save_path = r"D:\Study\Project\SignLAnguage\torch_data"

    # Initialize Mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2  # Lowered to reduce undetected hands
    )

    x_data, y_data = [], []
    not_detected_counts = Counter()
    minority_classes = ['nothing', 'M', 'N', 'space', 'del']

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        print(f"\nProcessing label: {label}")

        skipped = 0
        image_files = os.listdir(label_path)

        for img_file in tqdm(image_files, desc=f"  {label}", unit="img"):
            img_path = os.path.join(label_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Failed to load {img_path}")
                skipped += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                # Process original image
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    x_data.append(landmarks)
                    y_data.append(label)
                elif label == "nothing":
                    landmarks = [0.0] * 63  # Zero vector for no-hand images
                    x_data.append(landmarks)
                    y_data.append(label)
                else:
                    logging.info(f"No hand detected in {img_path}")
                    skipped += 1

                # Augment minority classes (1x for speed)
                if label in minority_classes:
                    aug_image = augment_image(image_rgb)
                    results = hands.process(aug_image)
                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]
                        landmarks = []
                        for lm in hand.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        x_data.append(landmarks)
                        y_data.append(label)
                    elif label == "nothing":
                        landmarks = [0.0] * 63  # Zero vector for no-hand images
                        x_data.append(landmarks)
                        y_data.append(label)
                    else:
                        logging.info(f"No hand detected in augmented {img_path}")
                        skipped += 1
            except Exception as e:
                logging.warning(f"Error processing {img_path}: {e}")
                skipped += 1

        not_detected_counts[label] = skipped

    hands.close()

    # Summary
    print("\n🧾 Summary of undetected hands per label:")
    for label, count in sorted(not_detected_counts.items()):
        print(f"  {label}: {count} image(s) with no hand detected")

    # Convert to torch tensors and save
    x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
    os.makedirs(save_path, exist_ok=True)
    torch.save(x_tensor, os.path.join(save_path, 'x_data.pt'))
    torch.save(y_data, os.path.join(save_path, 'y_data.pt'))

    print(f"\n✅ Saved:")
    print(f"  x_data.pt -> {os.path.join(save_path, 'x_data.pt')}")
    print(f"  y_data.pt -> {os.path.join(save_path, 'y_data.pt')}")
    print(f"  Total samples: {len(x_data)}")

if __name__ == "__main__":
    extract_landmarks()