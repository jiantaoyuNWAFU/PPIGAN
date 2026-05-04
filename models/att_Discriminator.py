import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_utils import select_device
import math
#from amino_acids import embedding_tensor


class Dis(nn.Module):
    def __init__(self, args):
        super(Dis, self).__init__()
        self.embedding_layer = nn.Embedding(21, args.em_dim, padding_idx=0)
        #self.embedding_layer.weight = nn.Parameter(embedding_tensor)
        self.drop_layer = nn.Dropout(args.sp_drop)
        self.conv_layers = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        for n in range(2, 35):
            if n <= 15:
                kernel_size = int(torch.ceil(torch.tensor(args.kernel_rate_1 * n ** 2)))
                stride = int(torch.ceil(torch.tensor(args.strides_rate_1 * (n - 1))))
                Lout = int(torch.ceil(torch.tensor((1500 - kernel_size + 1) / stride)))
                conv_layer = nn.Conv1d(in_channels=args.em_dim, out_channels=args.filter_num_1, kernel_size=kernel_size,
                                       padding=0, stride=stride)
            else:
                kernel_size = int(torch.ceil(torch.tensor(args.kernel_rate_2 * n ** 2)))
                stride = int(torch.ceil(torch.tensor(args.strides_rate_2 * (n - 1))))
                Lout = int(torch.ceil(torch.tensor((1500 - kernel_size + 1) / stride)))
                conv_layer = nn.Conv1d(in_channels=args.em_dim, out_channels=args.filter_num_2, kernel_size=kernel_size,
                                       padding=0, stride=stride)
            self.conv_layers.append(conv_layer)
            max_pool = nn.MaxPool1d(kernel_size=Lout)
            self.max_pools.append(max_pool)

        self.con_drop = nn.Dropout(args.con_drop)
        self.drop1 = nn.Dropout(args.fn_drop_1)
        self.drop2 = nn.Dropout(args.fn_drop_2)

        self.flatten = nn.Flatten()
        #self.attention = SelfAttention(args.node_num,args.node_num )

        self.dense1 = nn.Linear(150 * 14 + 175 * 19, args.node_num)
        self.rl = nn.ReLU()
        self.dense2 = nn.Linear(args.node_num, 2)

        self.softmax = nn.Softmax(dim=1)
        self.cbam = CBAM(args.filter_num_1)
        self.cbam1 = CBAM(args.filter_num_2)
        self.attention = SelfAttention(149,149)
        self.position_encoding = PositionalEncoding(args.em_dim, max_len=1500)

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def forward(self, input_a, input_b, is_gen):
        input_a = input_a.long()
        input_b = input_b.long()

        embedded_a = self.embedding_layer(input_a)
        #embedded_a = self.position_encoding(embedded_a)
        #print(embedded_a.type())
        #print(embedded_a.shape)
        masked_a = self.drop_layer(embedded_a.float())
        if torch.sum(is_gen).item() > 0:
            embedded_b = input_b
        else:
            embedded_b = self.embedding_layer(input_b)
            #embedded_b = self.position_encoding(embedded_b)
        #print(embedded_b.type())
        masked_b = self.drop_layer(embedded_b.float())
        #print(input_b.shape, masked_b.shape, is_gen)
        tensor = []
        #print(masked_a.shape)
        masked_a = masked_a.permute(0, 2, 1)
        masked_b = masked_b.permute(0, 2, 1)
        # print(masked_a.shape)
        for n in range(2, 35):
            conv_layer = self.conv_layers[n - 2]
            max_pool = self.max_pools[n - 2]

            conv_out_1 = conv_layer(masked_a)
            # print(conv_out_1.shape)
            conv_out_2 = conv_layer(masked_b)

            conv_out_1 = self.con_drop(conv_out_1)
            #print(conv_out_1.type)
            conv_out_2 = self.con_drop(conv_out_2)
            #print("conv_out1",conv_out_1.shape[1])
            #print("conv_out2", conv_out_2.shape[1])
            '''
            if n == 2:
                if conv_out_1.shape[1] == 150:
                    conv_out_1 = self.cbam(conv_out_1)
                    conv_out_2 = self.cbam(conv_out_2)
                else:
                    conv_out_1 = self.cbam1(conv_out_1)
                    conv_out_2 = self.cbam1(conv_out_2)
            
            if n==34:
                conv_out_1,_ = self.attention(conv_out_1)
                conv_out_2,_= self.attention(conv_out_2)
            '''
            pool_out_1 = max_pool(conv_out_1)
            pool_out_2 = max_pool(conv_out_2)
            #print(n)
            #print("poll",conv_out_2.shape)

            flat_out_1 = self.flatten(pool_out_1)
            flat_out_2 = self.flatten(pool_out_2)

            flat_out_1 = flat_out_1.view(-1, conv_layer.out_channels, 1)
            #print("flatten",flat_out_1.shape)
            #print("flatten", flat_out_2.shape)
            flat_out_2 = flat_out_2.view(-1, 1, conv_layer.out_channels)
            pool_out = torch.matmul(flat_out_1, flat_out_2)
            pool_out = 1 / 2 * (torch.max(pool_out, dim=1)[0] + torch.max(pool_out, dim=2)[0])
            # print(pool_out.shape)

            tensor.append(pool_out)


        concatenated = torch.cat(tensor, dim=-1)
        #print("concat",concatenated.shape)
        #self_attended_output, attention_weights = self.attention(concatenated)
        x = self.drop1(concatenated)
        # print(x.shape)
        x = self.dense1(x)
        #print(x.shape)
        x = self.drop2(x)
        x = self.rl(x)
        x = self.dense2(x)

        main_output = self.softmax(x)
        #
        #print(main_output)

        return main_output

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid_channel(avg_out + max_out)

        avg_max_pool = avg_out + max_out
        spatial_attention = torch.sigmoid(avg_max_pool)

        x = x * channel_attention * spatial_attention
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x       
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Query、Key、Value的线性变换层
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # 缩放系数
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 对输入进行线性变换，分别得到Query、Key、Value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # 计算Attention分数
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale_factor

        # 计算Attention权重
        attention_weights = torch.softmax(scores, dim=2)

        # 对Value进行加权求和
        attended_values = torch.bmm(attention_weights, value)

        return attended_values, attention_weights
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interaction_data',
                        default="./data/benchmarks/yeast core dataset from DeepFE-PPI/action_pair.tsv", type=str)
    parser.add_argument('--sequence_data',
                        default="./data/benchmarks/yeast core dataset from DeepFE-PPI/action_dictionary.tsv", type=str)
    parser.add_argument('--fold_index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=600, help='the maximum number of epochs')
    parser.add_argument('--outer_product', default=True, action='store_true',
                        help='Whether apply max-pooling on outer-product of two proteins')
    parser.add_argument('--cuda', default=True, action='store_true', help='Whether apply GPU to train the model')

    args = parser.parse_args()
    pair_file = args.interaction_data
    seq_file = args.sequence_data

    args.em_dim = 15
    args.sp_drop = 0.005
    args.kernel_rate_1 = 0.16
    args.strides_rate_1 = 0.15
    args.kernel_rate_2 = 0.14
    args.strides_rate_2 = 0.25
    args.filter_num_1 = 150
    args.filter_num_2 = 175
    args.con_drop = 0.05
    args.fn_drop_1 = 0.2
    args.fn_drop_2 = 0.1
    args.node_num = 256
    args.opti_switch = 1
    device = select_device("cuda:0")

    D = Dis(args).to(device)
    test_tensor1 = np.random.randint(21, size=(1, 1500))
    test_tensor2 = np.random.randint(21, size=(1, 1500))

    test_tensor1 = torch.from_numpy(test_tensor1).to(device)
    test_tensor2 = torch.from_numpy(test_tensor2).to(device)

    is_gen = torch.zeros((test_tensor1.size(0),)).to(device)
    test_res = D(test_tensor1, test_tensor2, is_gen)
    print(test_res)

'''
(None, 1500, 150)
(None, 1499, 150)
(None, 1498, 150)
(None, 1497, 150)
(None, 1495, 150)
(None, 1493, 150)
(None, 745, 150)
(None, 744, 150)
(None, 743, 150)
(None, 741, 150)
(None, 739, 150)
(None, 737, 150)
(None, 735, 150)
(None, 489, 150)
(None, 367, 175)
(None, 365, 175)
(None, 291, 175)
(None, 290, 175)
(None, 289, 175)
(None, 288, 175)
(None, 239, 175)
(None, 238, 175)
(None, 237, 175)
(None, 236, 175)
(None, 201, 175)
(None, 200, 175)
(None, 199, 175)
(None, 198, 175)
(None, 172, 175)
(None, 171, 175)
(None, 170, 175)
(None, 169, 175)
(None, 149, 175)
'''