import torch
import torch.nn.functional as F


def nearest_embedding_idx(embeddings, embedding_weights):
    distances = torch.cdist(embeddings, embedding_weights.unsqueeze(0), p=2)
    nearest_idxs = distances.argmin(dim=-1)
    return nearest_idxs


def convert_to_acid_ids_pytorch(fake_x, acid_embeddings, global_step):
    fake_to_display, distances = reverse_embedding_lookup_pytorch(acid_embeddings, fake_x)
    fake_to_display1, distances1 = reverse_embedding_lookup_pytorch1(acid_embeddings, fake_x)

    mean_cosine_distance = distances.mean().item()
    print(("Cosine_distance/FAKE", mean_cosine_distance, global_step))

    return fake_to_display, distances, fake_to_display1, distances1


def reverse_embedding_lookup_pytorch(acid_embeddings, embedded_sequence):
    acid_embeddings_expanded = acid_embeddings.unsqueeze(0)

    acid_embeddings_normalized = F.normalize(acid_embeddings_expanded, p=2, dim=2)
    embedded_sequence_normalized = F.normalize(embedded_sequence, p=2, dim=2)

    acid_embeddings_normalized = acid_embeddings_normalized.expand(embedded_sequence.size(0), -1, -1)

    emb_distances = torch.bmm(
        embedded_sequence_normalized,
        acid_embeddings_normalized.transpose(1, 2)
    )

    indices = torch.argmax(emb_distances, dim=2)
    max_distances = torch.max(emb_distances, dim=2).values

    return indices, max_distances


def reverse_embedding_lookup_pytorch1(acid_embeddings, embedded_sequence):
    acid_embeddings_expanded = acid_embeddings.unsqueeze(0)

    acid_embeddings = acid_embeddings_expanded.expand(embedded_sequence.size(0), -1, -1)

    acid_embeddings_normalized = F.normalize(acid_embeddings, p=2, dim=2)
    embedded_sequence_normalized = F.normalize(embedded_sequence, p=2, dim=2)

    emb_distances = torch.bmm(
        embedded_sequence_normalized,
        acid_embeddings_normalized.transpose(1, 2)
    )

    indices = torch.argmax(emb_distances, dim=2)
    max_distances = torch.max(emb_distances, dim=2).values

    return indices, max_distances