import torch
import torch.nn as nn
from Generator import Gen
from att_Discriminator import Dis
import numpy as np
import argparse

class cgan(nn.Module):
    def __init__(self, gen, dis):
        super(cgan, self).__init__()
        self.gen = gen
        self.dis = dis

    def forward(self,  input_seq,args):
        is_gen = torch.ones((input_seq.size(0),)).to(args.device)
        seq = input_seq.clone()
        z = torch.randint(0, 21, size=(input_seq.size(0), 1500, args.em_dim)).to(args.device)
        x = self.gen(seq, z)
        x = self.dis(seq, x, is_gen)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interaction_data', default="./data/benchmarks/yeast core dataset from DeepFE-PPI/action_pair.tsv", type=str)
    parser.add_argument('--sequence_data', default="./data/benchmarks/yeast core dataset from DeepFE-PPI/action_dictionary.tsv" ,type=str)
    parser.add_argument('--fold_index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=600, help='the maximum number of epochs')
    parser.add_argument('--outer_product', default=True, action='store_true', help='Whether apply max-pooling on outer-product of two proteins')
    parser.add_argument('--cuda', default=True, action='store_true', help='Whether apply GPU to train the model')

    args = parser.parse_args()
    pair_file = args.interaction_data
    seq_file = args.sequence_data

    args.em_dim=15
    args.sp_drop=0.005
    args.kernel_rate_1=0.16
    args.strides_rate_1=0.15
    args.kernel_rate_2=0.14
    args.strides_rate_2=0.25
    args.filter_num_1=150
    args.filter_num_2=175
    args.con_drop=0.05
    args.fn_drop_1=0.2
    args.fn_drop_2=0.1
    args.node_num=256
    args.opti_switch=1

    device = torch.device("cuda:0")
    
    D = Dis(args).to(device)
    G = Gen(args).to(device)

    cond_gan = cgan(G, D, args).to(device)
    test_tensor = torch.randint(0, 25, size=(2, 1500)).to(device)
    
    criterion = nn.CrossEntropyLoss()

    test_res = cond_gan(test_tensor, device)
    print(test_res)
    
    fake_labels = np.zeros((2, 2), dtype=int)
    fake_labels[:, 0] = 1
    fake_labels = torch.from_numpy(fake_labels).type(torch.float).to(device)

    fake_loss = criterion(test_res, fake_labels)

    fake_loss.backward()

    
    for name, param in cond_gan.named_parameters():
        print(name)

