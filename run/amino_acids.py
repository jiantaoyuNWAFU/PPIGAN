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
    id_to_amino_acid = {value: key for key, value in amino_acid.items()}
    sequence = ''.join(id_to_amino_acid.get(id, '?') for id in sequence_ids)
    return sequence

def print_acid_sequences(acid_sequence_ids, amino_acid_dict):
    seqs= []
    if isinstance(acid_sequence_ids, torch.Tensor):
        acid_sequence_ids = acid_sequence_ids.cpu().numpy()

    for seq_id in acid_sequence_ids:

        amino_acid_seq = ids_to_amino_acids(seq_id, amino_acid_dict)
        print(amino_acid_seq)
        seqs.append(amino_acid_seq)
    return seqs


def tsv_to_fasta(tsv_filename, fasta_filename):
    with open(tsv_filename, 'r') as tsv_file, open(fasta_filename, 'w') as fasta_file:
        for line in tsv_file:
            parts = line.strip().split('\t')
            if len(parts) < 2:  
                continue
            seq_id, sequence = parts[0], parts[1]

            fasta_file.write(f'>{seq_id}\n{sequence}\n')

def read_pretrain_amino(amino_file):
    embeddings = []
    with open(amino_file, 'r') as file:
        for line in file:
            embeddings.append([float(value) for value in line.strip().split()])

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
    return embeddings_tensor

embedding_tensor = torch.randn(20, 7)