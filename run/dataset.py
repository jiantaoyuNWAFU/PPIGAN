import torch
from torch.utils.data import Dataset
from input_preprocess import preprocess

class MyDataset(Dataset):
    def __init__(self, pair_file, seq_file):
        x_train_1, x_train_2, y_train, _ = preprocess(pair_file, seq_file)
        self.x_train_1 = x_train_1
        self.x_train_2 = x_train_2
        self.y_train = y_train

        self.pid1_list = []
        self.pid2_list = []
        with open(pair_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                self.pid1_list.append(parts[0])
                self.pid2_list.append(parts[1])

        assert len(self.pid1_list) == len(self.x_train_1), \
            f"Mismatch between number of pids and samples: pid={len(self.pid1_list)}, x={len(self.x_train_1)}"

    def __len__(self):
        return len(self.x_train_1)

    def __getitem__(self, index):
        x1 = torch.tensor(self.x_train_1[index], dtype=torch.float32)
        x2 = torch.tensor(self.x_train_2[index], dtype=torch.float32)
        y  = torch.tensor(self.y_train[index], dtype=torch.float32)

        pid1 = self.pid1_list[index]
        pid2 = self.pid2_list[index]

        return x1, x2, y, pid1, pid2