import torch
from torch.utils.data import Dataset
from input_preprocess import preprocess

class MyDataset(Dataset):
    def __init__(self, pair_file, seq_file):
        x_train_1, x_train_2, y_train, _ = preprocess(pair_file, seq_file)
        self.x_train_1 = x_train_1
        self.x_train_2 = x_train_2
        self.y_train = y_train
        # self.fix_text = fix_text

    def __len__(self):
        return len(self.x_train_1)

    def __getitem__(self, index):
        return self.x_train_1[index], self.x_train_2[index], self.y_train[index]
