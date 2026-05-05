import numpy as np


def preprocess(pair_file, seq_file):
    with open(pair_file, 'r') as f:

        lines = f.readlines()

    proteins_1 = [line.strip().split('\t')[0] for line in lines]
    proteins_2 = [line.strip().split('\t')[1] for line in lines]
    labels = [line.strip().split('\t')[2] for line in lines]
    protein_seq = {}

    with open(seq_file, 'r') as f:

        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].strip().split('\t')
        protein_seq[line[0]] = line[1]

    ID_TO_AMINO_ACID = {0: '0',
                        1: 'A',
                        2: 'C',
                        3: 'D',
                        4: 'E',
                        5: 'F',
                        6: 'G',
                        7: 'H',
                        8: 'I',
                        9: 'K',
                        10: 'L',
                        11: 'M',
                        12: 'N',
                        13: 'P',
                        14: 'Q',
                        15: 'R',
                        16: 'S',
                        17: 'T',
                        18: 'V',
                        19: 'W',
                        20: 'Y'}

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

    NON_STANDARD_AMINO_ACIDS = ['B', 'O', 'U', 'X', 'Z', 'J']

    k1 = []
    k2 = []
    k3 = []
    k_h = []

    for i in range(len(labels)):

        protein_1 = proteins_1[i]
        protein_2 = proteins_2[i]

        label = labels[i]

        seq_1 = protein_seq[protein_1]
        seq_2 = protein_seq[protein_2]

        a1 = np.zeros([1500, ], dtype=int)
        a2 = np.zeros([1500, ], dtype=int)

        k = 0
        for AA in seq_1:
            
            if k >= 1500:
                break
            
            if AA in amino_acid:
                a1[k] = amino_acid[AA]
            else:
                a1[k] = 0
                
            k += 1
        k1.append(a1)

        k = 0
        for AA in seq_2:
            if k >= 1500:
                break
            if AA in amino_acid:
                a2[k] = amino_acid[AA]
            else:
                a2[k] = 0
            k += 1
        k2.append(a2)

        if int(label) == 0:
            k3.append(np.array([1, 0]))
        else:
            k3.append(np.array([0, 1]))

        k_h.append(np.array([protein_1, protein_2]))

    m1 = np.stack(k1, axis=0)  
    m2 = np.stack(k2, axis=0)  
    m3 = np.stack(k3, axis=0)  
    m_h = np.stack(k_h, axis=0)  
    print(m1.shape)

    return m1, m2, m3, m_h

