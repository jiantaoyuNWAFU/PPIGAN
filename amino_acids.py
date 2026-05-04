import torch
import numpy as np
amino_acid = {'0': 0,
              'A': 1,
              'C': 2,
              'D': 3,
              'E': 4,
              'F': 5,
              'G': 6,
              'H': 7,
              'I': 8,
              'K': 9,
              'L': 10,
              'M': 11,
              'N': 12,
              'P': 13,
              'Q': 14,
              'R': 15,
              'S': 16,
              'T': 17,
              'V': 18,
              'W': 19,
              'Y': 20}

def ids_to_amino_acids(sequence_ids, amino_acid_dict):
    # 使用字典将ID序列转换为氨基酸序列
    id_to_amino_acid = {value: key for key, value in amino_acid.items()}
    sequence = ''.join(id_to_amino_acid.get(id, '?') for id in sequence_ids)
    return sequence

def print_acid_sequences(acid_sequence_ids, amino_acid_dict):
    # 转换为numpy数组（如果是PyTorch张量的话）
    seqs= []
    if isinstance(acid_sequence_ids, torch.Tensor):
        acid_sequence_ids = acid_sequence_ids.cpu().numpy()

    # 遍历每个序列并转换
    for seq_id in acid_sequence_ids:

        amino_acid_seq = ids_to_amino_acids(seq_id, amino_acid_dict)
        #print(seq_id)
        print(amino_acid_seq)
        seqs.append(amino_acid_seq)
    return seqs


def tsv_to_fasta(tsv_filename, fasta_filename):
    # 打开TSV文件和要写入的FASTA文件
    with open(tsv_filename, 'r') as tsv_file, open(fasta_filename, 'w') as fasta_file:
        # 遍历TSV文件的每一行
        for line in tsv_file:
            # 分割每行的内容为ID和序列
            parts = line.strip().split('\t')
            if len(parts) < 2:  # 确保行有足够的部分
                continue
            seq_id, sequence = parts[0], parts[1]

            # 将ID和序列格式化为FASTA格式，并写入FASTA文件
            fasta_file.write(f'>{seq_id}\n{sequence}\n')

def read_pretrain_amino(amino_file):
    # 将此路径替换为你的文件路径
    embeddings = []
    with open(amino_file, 'r') as file:
        for line in file:
            # 假设每行的嵌入值是用空格分隔的
            embeddings.append([float(value) for value in line.strip().split()])

    # 步骤2: 转换为Tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
    return embeddings_tensor
# 调用函数，转换TSV文件为FASTA文件
'''
def generator_seq(generate_seq, amino_acid):
    # np.squeeze 把维度为1的值去掉

    # 假设氨基酸字典为 amino_acids_dic
    # print(table)
    # 假设网络输出的概率矩阵为 output
    # print(prediction_seq)

    # 获取每个位点最大概率的氨基酸索引
    print(generate_seq.shape)
    amino_acids_index = np.argmax(generate_seq, axis=1)
    print(amino_acids_index)
    # print(amino_acids_index)
    # 根据氨基酸索引获取对应的氨基酸
    amino_acids_seq = []
    for row in amino_acids_index:
        seq = ''.join([amino_acid[index] for index in row])  # 将氨基酸索引转换为氨基酸序列
        amino_acids_seq.append(seq)

    # 输出氨基酸序列
    # print(amino_acids_seq)

    return amino_acids_seq
'''

#embedding_file_path = "/home/hlw/gengjing/deeptrio-pytorch1/embeddings/vec7_CTC.txt"
#embedding_tensor = read_pretrain_amino(embedding_file_path)
# 注释掉这一行
# embedding_tensor = read_pretrain_amino(embedding_file_path)

# 临时用随机 embedding（比如 20 种氨基酸，每个 7 维）

embedding_tensor = torch.randn(20, 7)
#print(embedding_tensor.shape)
#print(embedding_tensor.numpy())
#tsv_filename = 'D:\\gengjing\\exp\\deeptrio-master\\data\\benchmarks\\yeast core dataset from DeepFE-PPI\\action_dictionary.tsv'  # 这里替换为你的TSV文件路径
#fasta_filename = 'D:\\gengjing\\exp\\deeptrio-master\\data\\benchmarks\\yeast core dataset from DeepFE-PPI\\yeast.fasta'
#tsv_to_fasta(tsv_filename,fasta_filename)# 这里指定
#acid_sequence_ids1 = torch.tensor([[1, 1, 10, 14, 7], [8, 12, 5, 3, 2]])
#seqs = generator_seq(acid_sequence_ids1,amino_acid)
#print(seqs)
#print_acid_sequences(acid_sequence\
# _ids1,amino_acid)