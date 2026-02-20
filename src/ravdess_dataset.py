import os
import csv

id_to_emotion = {
1: "neutral",
2: "calm",
3: "happy",
4: "sad",
5: "angry",
6: "fearful",
7: "disgust",
8: "surprised",
}

emotion_counts = {
    "neutral": 0,
    "calm": 0,
    "happy": 0,
    "sad": 0,
    "angry": 0,
    "fearful": 0,
    "disgust": 0,
    "surprised": 0,
}

training_dataset = []


path = "data/ravdess"
ravdess = os.listdir(path)
for actor in ravdess:
        actor_path = path + "/" + actor
        files = os.listdir(actor_path)
        for i in range(len(files)):
            files_clean=files[i].replace(".wav", "")
            parts = files_clean.split("-")
            emotion_id = int(parts[2])
            actor_id = int(parts[6])
            emotion = id_to_emotion[emotion_id]
            emotion_counts[emotion] +=1
            full_path = str(actor_path) + "/" + str(files[i])
            training_dataset.append((full_path, emotion))
print(emotion_counts)
print(training_dataset[1])
print(len(training_dataset))

csv_path = "src/ravdess_index.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "emotion"])  # header
    for full_path, emotion in training_dataset:
        writer.writerow([full_path, emotion])

print("Saved CSV to:", csv_path)
      