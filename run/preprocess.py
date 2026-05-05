
import sys
import numpy as np
from tqdm import tqdm

def process_data(ds_file, id2seq_file, emb_file):
    id2index = {}
    seqs = []
    index = 0
    for line in open(id2seq_file):
        line = line.strip().split('\t')
        id2index[line[0]] = index
        seqs.append(line[1])
        index += 1
    seq_array = []
    id2_aid = {}
    sid = 0
    seq_size = 1500
    use_emb = 3
    hidden_dim = 25
    n_epochs = 50
    label_index = 2  
    sid1_index = 0
    sid2_index = 1
    if len(sys.argv) > 1:
        ds_file, label_index, rst_file, use_emb, hidden_dim, n_epochs = sys.argv[1:]
        label_index = int(label_index)
        use_emb = int(use_emb)  
        hidden_dim = int(hidden_dim)
        n_epochs = int(n_epochs)
    seq2t = s2t(emb_file)  
    max_data = -1
    limit_data = max_data > 0
    raw_data = []
    skip_head = True
    x = None
    count = 0
    for line in tqdm(open(ds_file)):
        if skip_head:
            skip_head = False
            continue
        line = line.rstrip('\n').rstrip('\r').split('\t')
        if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
            continue
        if id2_aid.get(line[sid1_index]) is None:
            id2_aid[line[sid1_index]] = sid
            sid += 1
            seq_array.append(seqs[id2index[line[sid1_index]]])
        line[sid1_index] = id2_aid[line[sid1_index]]
        if id2_aid.get(line[sid2_index]) is None:
            id2_aid[line[sid2_index]] = sid
            sid += 1
            seq_array.append(seqs[id2index[line[sid2_index]]])
        line[sid2_index] = id2_aid[line[sid2_index]]
        raw_data.append(line)
        if limit_data:
            count += 1
            if count >= max_data:
                break
    seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])
    print(seq_tensor[:10])
    seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
    seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

    class_map = {'0': 1, '1': 0}
    class_labels = np.zeros((len(raw_data), 2))
    for i in range(len(raw_data)):
        class_labels[i][class_map[raw_data[i][label_index]]] = 1.
    return raw_data, seq_tensor, seq_index1, seq_index2, class_labels