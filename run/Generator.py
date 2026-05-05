import argparse
import os
import warnings

import numpy as np
import torch
from torch import nn
from torch_utils import select_device
import torch.nn.functional as F

class Gen(nn.Module):
    def __init__(self, args, channel_num=1500):
        super(Gen, self).__init__()
        self.embedding = nn.Embedding(21, args.em_dim, padding_idx=0)
        self.fn = nn.Linear(args.em_dim, args.em_dim*10)
        self.softmax = nn.Softmax(dim=-1)
        self.conv1d_list = nn.ModuleList()
        self.conv1d_transpose = nn.Sequential()
        self.convtrans_list = nn.ModuleList()
        self.conv_num = args.conv_num
        self.channel_num = channel_num
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dense = nn.Linear(args.em_dim * 10-2 , 21)
        self.tanh = nn.Tanh()
        self.gen_Linear = nn.Linear(21, args.em_dim)
        c_list = [self.channel_num * 2]
        for i in range(self.conv_num):
            c = c_list[-1]
            print(c)
            self.conv1d_list.append(nn.Conv1d(c, c // 2, 3, 1, bias=False))
            self.conv1d_list.append(nn.BatchNorm1d(c // 2))
            c_list.append(c // 2)
        
        self.conv1d_transpose.add_module(f"conv2trans", nn.ConvTranspose1d(c_list[-1], c_list[-1], 3, padding=1, bias=False))
        self.conv1d_transpose.add_module(f"bn_conv2trans{i}", nn.BatchNorm1d(c_list[-1]))
        
        for i in range(self.conv_num - 1):
            c = c_list[self.conv_num - i]
            c_next = c_list[self.conv_num - i - 1]
            self.convtrans_list.append(nn.ConvTranspose1d(c, c_next, 3, bias=False))
            self.convtrans_list.append(nn.BatchNorm1d(c_next))
        
    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
    
    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def forward(self, input_a, input_b):
        input_a = input_a.long()
        input_b = input_b.long()

        embedded_a = self.embedding(input_a)
        x = torch.cat([embedded_a, input_b.float()], axis=1)
        x = self.fn(x)
        x_res = [x]
        for i in range(self.conv_num):
            cv1d = self.conv1d_list[i * 2]
            bn_cv = self.conv1d_list[i*2 + 1]
            x = cv1d(x)
            x = bn_cv(x)
            x = self.leakyRelu(x)
            x_res.append(x)
        x = self.conv1d_transpose(x)
        x = self.leakyRelu(x)
        for i in range(self.conv_num - 1):
            ct = self.convtrans_list[i*2]
            bn_ct = self.convtrans_list[i*2 + 1]
            x = ct(x + x_res.pop())
            x = bn_ct(x)
            x = self.leakyRelu(x)
        x = self.tanh(x + x_res.pop())
        x = self.dense(x)
        return x
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.embedding_layer = nn.Embedding(25, args.em_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
        )
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose1d(args.hidden_dim, args.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(args.hidden_dim, args.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(args.hidden_dim, args.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(args.hidden_dim, args.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(args.hidden_dim, args.em_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.tanh = nn.Tanh()

    def forward(self, input_a, input_b):
        input_a = input_a.long()
        input_b = input_b.long()

        embedded_a = self.embedding(input_a)
        x = torch.cat([embedded_a, input_b], axis=1)
        x = self.fc(x)
        x = x.unsqueeze(2)  
        x = self.conv_transpose(x)
        x = self.tanh(x)
        return x
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interaction_data', default="../data/benchmarks/yeast core dataset from DeepFE-PPI/action_pair.tsv", type=str)
    parser.add_argument('--sequence_data', default="../data/benchmarks/yeast core dataset from DeepFE-PPI/action_dictionary.tsv" ,type=str)
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
    args.conv_num = 10
    device = select_device("cuda:0")
    
    G = Gen(args).to(device)
    G.eval()
    test_tensor1 = np.random.randint(21, size=(1, 500))
    test_tensor2 = np.random.randint(21, size=(1, 500, args.em_dim))

    test_tensor1 = torch.from_numpy(test_tensor1).to(device)
    test_tensor2 = torch.from_numpy(test_tensor2).to(device)

    test_res = G(test_tensor1, test_tensor2)
    print(test_res)