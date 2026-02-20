import torch

def predict_knn(embedding, train_embeddings, k=1):
    distances = []

    for train_emb, emotion in train_embeddings:
        dist = torch.norm(embedding - train_emb)
        distances.append((dist, emotion))

    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]

    # majority vote
    votes = {}
    for _, emotion in top_k:
        votes[emotion] = votes.get(emotion, 0) + 1

    return max(votes, key=votes.get)