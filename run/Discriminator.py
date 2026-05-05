# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_utils import select_device
import math
import torch.nn.functional as F


class Dis(nn.Module):
    def __init__(self, args):
        super(Dis, self).__init__()
        self.embedding_layer = nn.Embedding(21, args.em_dim, padding_idx=0)
        self.drop_layer = nn.Dropout(args.sp_drop)
        self.conv_layers = nn.ModuleList()
        self.max_pools = nn.ModuleList()

        for n in range(2, 35):
            if n <= 15:
                kernel_size = int(torch.ceil(torch.tensor(args.kernel_rate_1 * n ** 2)))
                stride = int(torch.ceil(torch.tensor(args.strides_rate_1 * (n - 1))))
                Lout = int(torch.ceil(torch.tensor((1500 - kernel_size + 1) / stride)))
                conv_layer = nn.Conv1d(
                    in_channels=args.em_dim,
                    out_channels=args.filter_num_1,
                    kernel_size=kernel_size,
                    padding=0,
                    stride=stride
                )
            else:
                kernel_size = int(torch.ceil(torch.tensor(args.kernel_rate_2 * n ** 2)))
                stride = int(torch.ceil(torch.tensor(args.strides_rate_2 * (n - 1))))
                Lout = int(torch.ceil(torch.tensor((1500 - kernel_size + 1) / stride)))
                conv_layer = nn.utils.spectral_norm(
                    nn.Conv1d(
                        in_channels=args.em_dim,
                        out_channels=args.filter_num_2,
                        kernel_size=kernel_size,
                        padding=0,
                        stride=stride
                    )
                )
            self.conv_layers.append(conv_layer)
            self.max_pools.append(nn.MaxPool1d(kernel_size=Lout))

        self.con_drop = nn.Dropout(args.con_drop)
        self.drop1 = nn.Dropout(args.fn_drop_1)
        self.drop2 = nn.Dropout(args.fn_drop_2)
        self.cbam = CBAM(args.filter_num_1)
        self.cbam1 = CBAM(args.filter_num_2)
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(150 * 14 + 175 * 19, args.node_num)
        self.rl = nn.ReLU()
        self.dense2 = nn.Linear(args.node_num, 2)
        self.softmax = nn.Softmax(dim=1)

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def _ensure_3d(self, tensor):
        if tensor.dim() == 4:
            if 1 in tensor.shape:
                tensor = tensor.squeeze()
            else:
                tensor = torch.mean(tensor, dim=-1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        assert tensor.dim() == 3, f"Still not 3D after processing! Current dim: {tensor.dim()}, shape: {tensor.shape}"
        return tensor

    def forward(self, input_a, input_b, is_gen=None, return_logits: bool = False):
        input_a = input_a.long()
        embedded_a = self.embedding_layer(input_a)
        masked_a = self.drop_layer(embedded_a.float())
        masked_a = self._ensure_3d(masked_a)
        masked_a = masked_a.permute(0, 2, 1)

        if is_gen is not None and bool(is_gen):
            emb_weight = self.embedding_layer.weight  
            embedded_b = torch.matmul(input_b, emb_weight)  
        else:
            embedded_b = self.embedding_layer(input_b.long())

        masked_b = self.drop_layer(embedded_b.float())
        masked_b = self._ensure_3d(masked_b)
        masked_b = masked_b.permute(0, 2, 1)

        tensor = []
        for n in range(2, 35):
            conv_layer = self.conv_layers[n - 2]
            max_pool = self.max_pools[n - 2]
            conv_out_1 = self.con_drop(conv_layer(masked_a))
            conv_out_2 = self.con_drop(conv_layer(masked_b))
            pool_out_1 = max_pool(conv_out_1)
            pool_out_2 = max_pool(conv_out_2)
            if n == 10:
                if conv_out_1.shape[1] == 150:
                    conv_out_1 = self.cbam(conv_out_1)
                    conv_out_2 = self.cbam(conv_out_2)
                else:
                    conv_out_1 = self.cbam1(conv_out_1)
                    conv_out_2 = self.cbam1(conv_out_2)
            flat_out_1 = self.flatten(pool_out_1).view(-1, conv_layer.out_channels, 1)
            flat_out_2 = self.flatten(pool_out_2).view(-1, 1, conv_layer.out_channels)
            pool_out = torch.matmul(flat_out_1, flat_out_2)
            pool_out = 0.5 * (torch.max(pool_out, dim=1)[0] + torch.max(pool_out, dim=2)[0])
            tensor.append(pool_out)

        concatenated = torch.cat(tensor, dim=-1)
        x = self.drop1(concatenated)
        x = self.dense1(x)
        x = self.drop2(x)
        x = self.rl(x)
        x = self.dense2(x)
        if return_logits:
            return x
        return self.softmax(x)


class Dis_rcnn(nn.Module):
    def __init__(self, args):
        super(Dis_rcnn, self).__init__()
        self.seq_size = 1500
        self.dim = args.em_dim
        self.hidden_dim = args.hidden_dim
        self.embedding_layer = nn.Embedding(21, args.em_dim, padding_idx=0)
        self.l1 = nn.Conv1d(self.dim, self.hidden_dim, 3)
        self.r1 = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.l2 = nn.Conv1d(75, self.hidden_dim, 3)
        self.r2 = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.l3 = nn.Conv1d(75, self.hidden_dim, 3)
        self.r3 = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.l4 = nn.Conv1d(75, self.hidden_dim, 3)
        self.r4 = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.l5 = nn.Conv1d(75, self.hidden_dim, 3)
        self.r5 = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.l6 = nn.Conv1d(75, self.hidden_dim, 3)
        self.maxpool = nn.MaxPool1d(3)
        self.global_avepool = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.Linear(self.hidden_dim, 100)
        self.lrelu1 = nn.LeakyReLU(0.3)
        self.dense2 = nn.Linear(100, int((self.hidden_dim + 7) / 2))
        self.lrelu2 = nn.LeakyReLU(0.3)
        self.dense3 = nn.Linear(16, 2)

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def forward(self, input_a, input_b, is_gen=None):
        input_a = input_a.long()
        input_b = input_b.long()
        embedded_a = self.embedding_layer(input_a)
        embedded_b = self.embedding_layer(input_b)
        input_a = embedded_a.permute(0, 2, 1).float()
        s1 = self.maxpool(self.l1(input_a))
        s1 = s1.permute(0, 2, 1)
        s1_r1, _ = self.r1(s1)
        s1 = torch.cat((s1_r1, s1), dim=2).permute(0, 2, 1)
        s1 = self.maxpool(self.l2(s1))
        s1_r2, _ = self.r2(s1.permute(0, 2, 1))
        s1 = torch.cat((s1_r2, s1.permute(0, 2, 1)), dim=2).permute(0, 2, 1)
        s1 = self.maxpool(self.l3(s1))
        s1_r3, _ = self.r3(s1.permute(0, 2, 1))
        s1 = torch.cat((s1_r3, s1.permute(0, 2, 1)), dim=2).permute(0, 2, 1)
        s1 = self.maxpool(self.l4(s1))
        s1_r4, _ = self.r4(s1.permute(0, 2, 1))
        s1 = torch.cat((s1_r4, s1.permute(0, 2, 1)), dim=2).permute(0, 2, 1)
        s1 = self.maxpool(self.l5(s1))
        s1_r5, _ = self.r5(s1.permute(0, 2, 1))
        s1 = torch.cat((s1_r5, s1.permute(0, 2, 1)), dim=2)
        s1 = self.l6(s1.permute(0, 2, 1))
        s1 = self.global_avepool(s1).squeeze()

        input_b = embedded_b.permute(0, 2, 1).float()
        s2 = self.maxpool(self.l1(input_b))
        s2 = s2.permute(0, 2, 1)
        s2_r1, _ = self.r1(s2)
        s2 = torch.cat((s2_r1, s2), dim=2).permute(0, 2, 1)
        s2 = self.maxpool(self.l2(s2))
        s2_r2, _ = self.r2(s2.permute(0, 2, 1))
        s2 = torch.cat((s2_r2, s2.permute(0, 2, 1)), dim=2).permute(0, 2, 1)
        s2 = self.maxpool(self.l3(s2))
        s2_r3, _ = self.r3(s2.permute(0, 2, 1))
        s2 = torch.cat((s2_r3, s2.permute(0, 2, 1)), dim=2).permute(0, 2, 1)
        s2 = self.maxpool(self.l4(s2))
        s2_r4, _ = self.r4(s2.permute(0, 2, 1))
        s2 = torch.cat((s2_r4, s2.permute(0, 2, 1)), dim=2).permute(0, 2, 1)
        s2 = self.maxpool(self.l5(s2))
        s2_r5, _ = self.r5(s2.permute(0, 2, 1))
        s2 = torch.cat((s2_r5, s2.permute(0, 2, 1)), dim=2)
        s2 = self.l6(s2.permute(0, 2, 1))
        s2 = self.global_avepool(s2).squeeze()

        merge_text = torch.mul(s1, s2)
        x = self.dense1(merge_text)
        x = self.lrelu1(x)
        x = self.dense2(x)
        x = self.lrelu2(x)
        x = self.dense3(x)
        return F.softmax(x, dim=1)


class Dis_cnn(nn.Module):
    def __init__(self, args):
        super(Dis_cnn, self).__init__()
        self.embedding_layer = nn.Embedding(21, args.em_dim, padding_idx=0)
        self.l1 = nn.Conv1d(in_channels=args.em_dim, out_channels=args.hidden_dim, kernel_size=3)
        self.l2 = nn.Conv1d(in_channels=args.hidden_dim, out_channels=args.hidden_dim, kernel_size=3)
        self.l3 = nn.Conv1d(in_channels=args.hidden_dim, out_channels=args.hidden_dim, kernel_size=3)
        self.l4 = nn.Conv1d(in_channels=args.hidden_dim, out_channels=args.hidden_dim, kernel_size=3)
        self.l5 = nn.Conv1d(in_channels=args.hidden_dim, out_channels=args.hidden_dim, kernel_size=3)
        self.l6 = nn.Conv1d(in_channels=args.hidden_dim, out_channels=args.hidden_dim, kernel_size=3)
        self.maxpool = nn.MaxPool1d(2)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.dense2 = nn.Linear(args.hidden_dim, int((args.hidden_dim + 7) / 2))
        self.dense3 = nn.Linear(int((args.hidden_dim + 7) / 2), 2)

    def _ensure_3d(self, tensor):
        if tensor.dim() == 4:
            if 1 in tensor.shape:
                tensor = tensor.squeeze()
            else:
                tensor = torch.mean(tensor, dim=-1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        assert tensor.dim() == 3, f"Still not 3D after processing! Current dim: {tensor.dim()}, shape: {tensor.shape}"
        return tensor

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def forward(self, input_a, input_b, is_gen=None):
        input_a = input_a.long()
        embedded_a = self.embedding_layer(input_a)
        embedded_a = self._ensure_3d(embedded_a)

        if is_gen is None:
            use_generated = False
        elif torch.is_tensor(is_gen):
            use_generated = torch.sum(is_gen).item() > 0
        else:
            use_generated = bool(is_gen)

        if use_generated:
            embedded_b = self._ensure_3d(input_b)
        else:
            input_b = input_b.long()
            embedded_b = self.embedding_layer(input_b)
            embedded_b = self._ensure_3d(embedded_b)

        s1 = self.l1(embedded_a.float().permute(0, 2, 1))
        s1 = self.maxpool(s1)
        s1 = self.l2(s1)
        s1 = self.maxpool(s1)
        s1 = self.l3(s1)
        s1 = self.maxpool(s1)
        s1 = self.l4(s1)
        s1 = self.maxpool(s1)
        s1 = self.l5(s1)
        s1 = self.maxpool(s1)
        s1 = self.l6(s1)
        s1 = self.global_avgpool(s1).squeeze()

        s2 = self.l1(embedded_b.float().permute(0, 2, 1))
        s2 = self.maxpool(s2)
        s2 = self.l2(s2)
        s2 = self.maxpool(s2)
        s2 = self.l3(s2)
        s2 = self.maxpool(s2)
        s2 = self.l4(s2)
        s2 = self.maxpool(s2)
        s2 = self.l5(s2)
        s2 = self.maxpool(s2)
        s2 = self.l6(s2)
        s2 = self.global_avgpool(s2).squeeze()

        merge_text = s1 * s2
        x = self.dense1(merge_text)
        x = self.leaky_relu(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dense3(x)
        return x 


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
