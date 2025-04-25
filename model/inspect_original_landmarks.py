# inspect_original_landmarks.py
import pandas as pd

csv_path = r"D:\Study\Project\SignLanguageRecognition\data\original_landmarks.csv"
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit()

print(f"Rows: {len(data)}")
print("\nColumns:", list(data.columns))
print("\nLabel counts:\n", data['label'].value_counts())
print("\nNumber of NaN values per column:\n", data.isna().sum())