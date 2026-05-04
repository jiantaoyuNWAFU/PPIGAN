import argparse
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch_utils import select_device
from Discriminator import Dis
from Generator import Gen
from CGAN import cgan
from dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
from amino_acids import amino_acid, ids_to_amino_acids, print_acid_sequences
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from input_preprocess import preprocess
import sys
from collections import defaultdict
#from tsne import visualize_vector_distance
'''
if 'D:\\gengjing\\exp\\deeptrio-master\\deeptrio-pytorch\\embeddings' not in sys.path:
    sys.path.append('D:\\gengjing\\exp\\deeptrio-master\\deeptrio-pytorch\\embeddings')
'''


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    print(train_size, test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
    
def calculate_protein_degree(tsv_path):
    """计算蛋白质度数字典"""
    degree_dict = defaultdict(int)
    with open(tsv_path) as f:
        for line in f:
            p1, p2, label = line.strip().split('\t')
            if label == '1':  # 仅统计正例
                degree_dict[p1] += 1
                degree_dict[p2] += 1
    return degree_dict

def train(args):
    if args.train_dataset == "" and args.test_dataset == "":
        # raw_data, seq_tensor, seq_index1, seq_index2, y_train = process_data(args.interaction_data, args.sequence_data, args.emb_file)
        dataset = MyDataset(args.interaction_data, args.sequence_data)
        # print(seq_index1)
        train_dataset, test_dataset = split_dataset(dataset)
        torch.save(train_dataset, "./data/train_dataset.pth")
        torch.save(test_dataset, "./data/test_dataset.pth")
    else:
        train_dataset = torch.load(args.train_dataset)
        test_dataset = torch.load(args.test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    D = Dis(args)  # 初始化
    G = Gen(args)
    G.weight_init(mean=0.0, std=0.02)
    if args.d_pth == "":
        D.weight_init(mean=0.0, std=0.02)
        args.is_only_dis = False
    else:
        sd = torch.load(args.d_pth)
        D.load_state_dict(sd)
        args.is_only_dis = False

    G.to(args.device)  # 移动到设备上
    D.to(args.device)
    cond_gan = cgan(G, D).to(args.device)

    # 训练过程
    criterion = nn.BCELoss()
    criterion_gen = nn.BCELoss()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-6)
    optimizer_cond_gan = torch.optim.Adam(cond_gan.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-6)

    best_acc, best_epoch = 0, 0
    torch.autograd.set_detect_anomaly(True)
    global_step = 0
    g_losses = []
    d_losses = []
    for epoch in range(args.epoch):
        D.train()
        G.train()
        for i, (x1, x2, y) in enumerate(train_loader):

            x1 = x1.to(args.device)
            # print(x1.shape)
            x2 = x2.to(args.device)
            y = y.type(torch.float).to(args.device)

            # 训练判别器
            optimizer_D.zero_grad()

            is_gen = torch.zeros((x1.size(0),)).to(args.device)
            real_outputs = D(x1, x2, is_gen)
            # print(real_outputs, y)
            real_loss = criterion(real_outputs, y)  # 判别器在真实样本上的损失
            # only_discriminator
            if args.is_only_dis:
                real_loss.backward()
                optimizer_D.step()
                print(f"Epoch [{epoch + 1}/{args.epoch}], Step [{i + 1}/{len(train_loader)}], "
                      f"D_loss: {real_loss.item():.4f}")
                continue
            true_index = (y == torch.tensor([0, 1], device=args.device)).all(dim=1)
            # print(x1.shape)
            x1 = x1[true_index]  # 蛋白质A
            x2 = x2[true_index]  # 蛋白质B
            y = y[true_index]  # label
            # print(x1.shape)
            '''
            random_number = np.random.rand()
            
            # 如果随机数小于0.5，则选择蛋白质A作为生成器的条件；否则选择蛋白质B
            if random_number < 0.5:
                chosen_protein = x1
            else:
                chosen_protein = x2
            '''
            protein_degrees = calculate_protein_degree(args.interaction_data)
            #print(protein_degrees)
            degree_x1 = protein_degrees.get(x1, 0)
            
    
            degree_x2 = protein_degrees.get(x2, 0)
    
            # 优先选择度数更高的蛋白质
            if degree_x1 > degree_x2:
                chosen_protein = x1
            elif degree_x2 > degree_x1:
                chosen_protein = x2
            else:
            # 如果度数相同，则随机选择
                if np.random.rand() < 0.5:
                    chosen_protein = x1
                else:
                    chosen_protein = x2

            fake_labels = np.zeros((x1.size(0), 2), dtype=int)
            fake_labels[:, 0] = 1
            fake_labels = torch.from_numpy(fake_labels).type(torch.float).to(args.device)
            
            real_labels = np.zeros((x1.size(0), 2), dtype=int)
            real_labels[:, 1] = 1
            real_labels = torch.from_numpy(real_labels).type(torch.float).to(args.device)

            z = np.random.randint(21, size=(x1.size(0), 1500, args.em_dim))
            z = torch.from_numpy(z).to(args.device)

            is_gen = torch.ones((x2.size(0),)).to(args.device)
            fake_inputs = G(chosen_protein, z)
            #print(fake_inputs)
            # print(fake_inputs.detach().type())#生成器输入为蛋白质A和一个随机向量
            # print("fake_inputs:",fake_inputs)
            
            fake_outputs = D(chosen_protein, fake_inputs, is_gen)  # 生成样本的标签为[1,1]
           
            # embedding_weights_D = D.embedding_layer.weight
            # print(embedding_weights_D)
            #embedding_weights_G = G.embedding.weight
            # print(embedding_weights_G)
            # print("embedding:",embedding_weights.shape)

            # seq_ids = nearest_embedding_idx(fake_inputs,embedding_weights)
            
            #acid_sequence_ids, distances, acid_sequence_ids1, distances1 = convert_to_acid_ids_pytorch(fake_inputs,
            #                                                                                           embedding_weights_G,
            #                                                                                           global_step,
            #                                                                                           )
            # acid_sequence_ids1 = torch.tensor([[0, 1, 10, 14, 7], [8, 12, 5, 3, 2]])

            # seqs = generator_seq(fake_inputs.detach().cpu().numpy(),amino_acid)
            #seqs = print_acid_sequences(acid_sequence_ids1, amino_acid)
            '''
            with open('generated_sequences.fasta', 'w') as fasta_file:
                for idx, seq in enumerate(seqs):
                    # 转换ID序列为氨基酸序列
                    fasta_file.write(f'>Sequence_{idx}\n{seq}\n')
            '''
            # acid_sequence_ids = reverse_embedding_lookup_pytorch(fake_outputs, embedding_weights)
            # acid_sequences = [index_to_acid[idx.item()] for idx in ac]
            # print(acid_sequence_ids)
            fake_loss_a = criterion(fake_outputs, fake_labels)

            # fake_inputs = G(x2, z)
            # fake_outputs = D(x2, fake_inputs.detach(), is_gen)
            # fake_loss_b = criterion(fake_outputs, fake_labels)

            d_loss = args.beta_real_loss * real_loss + fake_loss_a * args.beta_fake_loss
            d_losses.append(d_loss.item())
            d_loss.backward()
            # for name, param in D.named_parameters():
            #     print(name, param.grad)
            optimizer_D.step()
            global_step += 1
            # 训练生成器
            for k in range(2):
              optimizer_cond_gan.zero_grad()
              fake_outputs_a = cond_gan(chosen_protein, args)
              # fake_outputs_b = cond_gan(x2, args)
              # fake_outputs = D(x1, fake_inputs, is_gen)
              g_loss = criterion_gen(fake_outputs_a, real_labels)
              g_losses.append(g_loss.item())
              # g_loss = (criterion_gen(fake_outputs_a, real_labels) + criterion(fake_outputs_b, real_labels)) * 0.5
              # 固定参数
              for name, param in cond_gan.dis.named_parameters():
                param.requires_grad = False

              for name, param in cond_gan.gen.named_parameters():
                param.requires_grad = True
              g_loss.backward()
              '''
              if global_step%50 == 0:
                x1_embed =G.embedding(x1.long())
                x2_embed = G.embedding(x2.long())
                print(x2_embed.shape)
                visualize_vector_distance(fake_inputs[0].cpu().detach().numpy(),x2_embed[0].cpu().detach().numpy())
              '''
              # 打印各层的梯度
              optimizer_cond_gan.step()
              # print("after",embedding_weights)
              for param in cond_gan.dis.parameters():
                param.requires_grad = True
              #embedding_weights = cond_gan.gen.embedding.weight
              # print("after", embedding_weights)
              print(f"Epoch [{epoch + 1}/{args.epoch}], Step [{i + 1}/{len(train_loader)}], "
                  f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")
              #output_interval = 2
            
              #print(seqs)
            

        # 验证模型
        D.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for x1, x2, y in test_loader:
                x1 = x1.to(args.device)
                # print(x1.shape)
                x2 = x2.to(args.device)
                y = y.type(torch.float).to(args.device)

                is_gen = torch.zeros((x1.size(0),)).to(args.device)
                outputs = D(x1, x2, is_gen)
                # 将outputs和y转换为numpy数组
                outputs = outputs.cpu().numpy()[:, 1]
                y = y.cpu().numpy()[:, 1]
                outputs = (outputs > 0.5).astype(int)

                y_true.extend(y.tolist())
                y_pred.extend(outputs.tolist())

            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            # 计算性能指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            if args.is_only_dis:
                torch.save(D.state_dict(),os.path.join('./models/only_dis/human_biogrid/', f'D_best_{epoch}_gen(x,z)_deeprio_degree_5.pth'))
            else:
                torch.save(D.state_dict(),os.path.join('./models/dis_after_gen/human_biogrid/', f'D_best_{epoch}_{accuracy}_gen(x,z)_deeptrio_degree_0610.pth'))
                torch.save(G.state_dict(),os.path.join('./models/dis_after_gen/human_biogrid/', f'G_best_{epoch}_{accuracy}_gen(x,z)_deeptrio_degree_0610.pth'))
                print("Best model saved!")
            print("混淆矩阵:")
            print(cm)
            print("准确率:", accuracy)
            print("精确率:", precision)
            print("特异性:", specificity)
            print("召回率:", recall)
            print("F1值:", f1)
            print("MCC:", mcc)
            #print(f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}")
            print("===============================================")
            if args.is_only_dis:
                with open(f"./log/human_biogrid/log_only_dis_deeptrio_15_40_0610.txt", "a+") as f:
                    f.write(
                        f"Epoch [{epoch + 1}/{args.epoch}] \ncm:{cm} \nAccuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, MCC:{mcc}\n")
                    f.write(f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}\n")
                    f.write("===============================================\n")
                continue
            with open(f"./log/human_biogrid/log_{args.beta_real_loss}_{args.beta_fake_loss}_deeptrio_0610.txt", "a+") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{args.epoch}] \ncm:{cm} \nAccuracy: {accuracy}, Precision: {precision},Specificity: {specificity}, Recall: {recall}, F1: {f1}, MCC:{mcc}\n")
                f.write(f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}\n")
                f.write("===============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ✅ 1) 把 interaction_data / sequence_data 变成“参数”，写在 parser 里
    parser.add_argument('--interaction_data', type=str,
                        default=r"D:\QQ\qq文件\PPIGAN\PPIGAN\Dataset\Biogrid-human\third_human_MV_pair.tsv")
    parser.add_argument('--sequence_data', type=str,
                        default=r"D:\QQ\qq文件\PPIGAN\PPIGAN\Dataset\Biogrid-human\double_human_MV_database.tsv")

    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--train_dataset', default="", type=str)
    parser.add_argument('--test_dataset', default="", type=str)
    parser.add_argument('--d_pth', default="", type=str)
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--beta_real_loss', default=0.5, type=float)
    parser.add_argument('--beta_fake_loss', default=0.5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--conv_num', default=10, type=int)
    parser.add_argument('--em_dim', default=15, type=int)
    parser.add_argument('--hidden_dim', default=25, type=int)
    parser.add_argument('--sp_drop', default=0.005, type=float)
    parser.add_argument('--kernel_rate_1', default=0.16, type=float)
    parser.add_argument('--strides_rate_1', default=0.15, type=float)
    parser.add_argument('--kernel_rate_2', default=0.14, type=float)
    parser.add_argument('--strides_rate_2', default=0.25, type=float)
    parser.add_argument('--filter_num_1', default=150, type=int)
    parser.add_argument('--filter_num_2', default=175, type=int)
    parser.add_argument('--con_drop', default=0.05, type=float)
    parser.add_argument('--fn_drop_1', default=0.2, type=float)
    parser.add_argument('--fn_drop_2', default=0.1, type=float)
    parser.add_argument('--node_num', default=256, type=int)

    args = parser.parse_args()

    args.device = select_device("cuda:0" if args.cuda else "cpu")

    args.d_pth = ""
    args.epoch = 50

    train(args)


