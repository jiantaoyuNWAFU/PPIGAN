import torch
import torch.nn.functional as F



def nearest_embedding_idx(embeddings, embedding_weights):
    # embeddings: 生成的嵌入向量，形状为 [batch_size, seq_len, embedding_dim]
    # embedding_weights: 预训练的嵌入权重，形状为 [vocab_size, embedding_dim]
    distances = torch.cdist(embeddings, embedding_weights.unsqueeze(0), p=2)  # 计算距离
    nearest_idxs = distances.argmin(dim=-1)  # 找到最近嵌入的索引
    return nearest_idxs


def convert_to_acid_ids_pytorch(fake_x, acid_embeddings,  global_step ):
    # 假设 fake_x 已经是正确形状的张量
    fake_to_display, distances = reverse_embedding_lookup_pytorch(acid_embeddings, fake_x)
    fake_to_display1, distances1 = reverse_embedding_lookup_pytorch1(acid_embeddings, fake_x)
    # 使用TensorBoard记录信息，如果未提供writer实例，则创建一个

    # 记录平均余弦距离
    mean_cosine_distance = distances.mean().item()
    #print("Cosine_distance/FAKE", mean_cosine_distance, global_step)
    print(("Cosine_distance/FAKE", mean_cosine_distance, global_step))
    # 在PyTorch中，不需要特别的squeeze操作，除非需要移除大小为1的维度
    return fake_to_display, distances,fake_to_display1, distances1
def reverse_embedding_lookup_pytorch(acid_embeddings, embedded_sequence):
    # 假设:
    # acid_embeddings 是一个形状为 [num_embeddings, embedding_dim] 的二维张量
    # embedded_sequence 是一个形状为 [batch_size, seq_len, embedding_dim] 的三维张量

    # 首先，我们不需要转置 embedded_sequence，但我们需要确保 acid_embeddings 的形状适配
    # 为了与 embedded_sequence 的最后一个维度进行点乘，我们将 acid_embeddings 视为 [1, embedding_dim, num_embeddings]
    acid_embeddings_expanded = acid_embeddings.unsqueeze(0)

    # 正则化嵌入向量和输入序列
    acid_embeddings_normalized = F.normalize(acid_embeddings_expanded, p=2, dim=2)
    embedded_sequence_normalized = F.normalize(embedded_sequence, p=2, dim=2)

    # 使用batch matrix multiplication进行计算，因此我们需要扩展acid_embeddings到batch_size
    acid_embeddings_normalized = acid_embeddings_normalized.expand(embedded_sequence.size(0), -1, -1)

    # 计算嵌入向量之间的余弦相似度
    emb_distances = torch.bmm(embedded_sequence_normalized, acid_embeddings_normalized.transpose(1, 2))

    # 找到最相似嵌入向量的索引
    indices = torch.argmax(emb_distances, dim=2)
    max_distances = torch.max(emb_distances, dim=2).values

    return indices, max_distances


def reverse_embedding_lookup_pytorch1(acid_embeddings, embedded_sequence):
    # 假设:
    # acid_embeddings 是一个形状为 [num_embeddings, embedding_dim] 的二维张量
    # embedded_sequence 是一个形状为 [batch_size, seq_len, embedding_dim] 的三维张量

    # 首先，我们不需要转置 embedded_sequence，但我们需要确保 acid_embeddings 的形状适配
    # 为了与 embedded_sequence 的最后一个维度进行点乘，我们将 acid_embeddings 视为 [1, embedding_dim, num_embeddings]
    acid_embeddings_expanded = acid_embeddings.unsqueeze(0)
    # 正则化嵌入向量和输入序列

    # 使用batch matrix multiplication进行计算，因此我们需要扩展acid_embeddings到batch_size
    acid_embeddings = acid_embeddings_expanded.expand(embedded_sequence.size(0), -1, -1)

    acid_embeddings_normalized = F.normalize(acid_embeddings, p=2, dim=2)
    embedded_sequence_normalized = F.normalize(embedded_sequence, p=2, dim=2)

    # 计算嵌入向量之间的余弦相似度
    emb_distances = torch.bmm(embedded_sequence_normalized, acid_embeddings_normalized.transpose(1, 2))

    # 找到最相似嵌入向量的索引
    indices = torch.argmax(emb_distances, dim=2)
    max_distances = torch.max(emb_distances, dim=2).values

    return indices, max_distances




