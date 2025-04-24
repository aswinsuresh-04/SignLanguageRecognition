import os
from collections import Counter

data_dir = r"D:\Study\Project\SignLAnguage\asl_alphabet_train\asl_alphabet_train"
label_counts = Counter()
for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if os.path.isdir(label_path):
        label_counts[label] = len(os.listdir(label_path))
print("Image Counts per Class:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")