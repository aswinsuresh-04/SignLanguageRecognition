import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,  # Lowered to improve detection
    min_tracking_confidence=0.5
)

def main():
    data_dir = r"D:\Study\Project\SignLAnguage\dataset"
    label = "nothing"  # Change to 'P', 'C', 'Q', 'M', 'N', etc.
    label_dir = os.path.join(data_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Clear existing images to avoid duplicates
    for file in os.listdir(label_dir):
        os.remove(os.path.join(label_dir, file))
    
    cap = cv2.VideoCapture(0)
    sample_count = 0
    max_samples = 3000

    while sample_count < max_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if (results.multi_hand_landmarks and label != "nothing") or (label == "nothing"):
            cv2.imwrite(os.path.join(label_dir, f"{label}_{sample_count}.jpg"), frame)
            sample_count += 1
            cv2.putText(frame, f"Samples: {sample_count}/{max_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()