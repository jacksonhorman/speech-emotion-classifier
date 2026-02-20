import torch

def build_centroids(embeddings_by_emotion):
    emotion_centroids = {}
    for emotion, emb_list in embeddings_by_emotion.items():
        stacked = torch.stack(emb_list)
        emotion_centroids[emotion] = stacked.mean(dim=0)
    return emotion_centroids


def predict_emotions(embedding, emotion_centroids):
    best_emotion = None
    best_distance = None

    for emotion, centroid in emotion_centroids.items():
        distance = torch.norm(embedding - centroid)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_emotion = emotion
    return best_emotion