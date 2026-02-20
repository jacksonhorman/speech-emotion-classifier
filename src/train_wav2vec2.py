import csv
import torch
from extract_embedding import extract_embeddings
from centroid_classifier import build_centroids, predict_emotions
from neighbor_classifier import predict_knn


csv_path = "src/ravdess_index.csv"
N = 500        
K = 1            

embeddings_by_emotion = {}
train_embeddings = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i == N:
            break

        audio = row["path"]
        emotion = row["emotion"]
        print(f"Processing {i+1}/",N,":", emotion)

        embedding = extract_embeddings(audio)

        if emotion not in embeddings_by_emotion:
            embeddings_by_emotion[emotion] = []

        embeddings_by_emotion[emotion].append(embedding)
        train_embeddings.append((embedding, emotion))


print("Done.")
print({k: len(v) for k, v in embeddings_by_emotion.items()})


emotion_centroids = build_centroids(embeddings_by_emotion)

print("\nTesting ONE sample per emotion:")

for emotion, emb_list in embeddings_by_emotion.items():
    test_emb = emb_list[0]

    centroids_no_self = {}
    for emo2, list2 in embeddings_by_emotion.items():
        if emo2 == emotion:
            train_list = list2[1:]
            if len(train_list) == 0:
                continue
            centroids_no_self[emo2] = torch.stack(train_list).mean(dim=0)
        else:
            centroids_no_self[emo2] = torch.stack(list2).mean(dim=0)

    pred_centroid = predict_emotions(test_emb, centroids_no_self)

    train_embeddings_no_self = []
    for emb, emo in train_embeddings:
        if emo == emotion and torch.equal(emb, test_emb):
            continue
        train_embeddings_no_self.append((emb, emo))

    pred_knn = predict_knn(test_emb, train_embeddings_no_self, k=K)

    print(
        f"True: {emotion:9s} | "
        f"Centroid: {pred_centroid:9s} | "
        f"kNN(k={K}): {pred_knn:9s}"
    )