import argparse
import os
import numpy as np
import torch
from torch import nn
from torch_utils import select_device
from Discriminator import Dis
from Generator import Gen
from CGAN import cgan
from dataset import MyDataset
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from amino_acids import amino_acid, print_acid_sequences,print_acid_sequences, tsv_to_fasta
from helper import convert_to_acid_ids_pytorch  # 导入转换函数
from collections import defaultdict
id_to_amino = {v: k for k, v in amino_acid.items()}

def nearest_embedding_idx(embeddings, embedding_weights):
    """
    计算嵌入向量与嵌入权重的最近邻，返回对应的氨基酸ID
    embeddings: 生成器输出的嵌入向量，形状为 [batch_size, seq_len, em_dim]
    embedding_weights: 嵌入层权重，形状为 [21, em_dim]
    """
    # 计算余弦相似度：(batch, seq_len, 21)
    similarity = torch.matmul(embeddings, embedding_weights.t())
    # 取相似度最大的索引（氨基酸ID）
    return torch.argmax(similarity, dim=-1)

def save_generated_seqs(seqs, epoch, step, fold_index, save_dir="./generated_seqs"):
    """保存生成的氨基酸序列到文件"""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"fold_{fold_index}_epoch_{epoch}_step_{step}.txt")
    with open(file_path, "a") as f:
        for seq in seqs:
            f.write(seq + "\n")  # 每个序列占一行
            
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
  
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    print(train_size, test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
    
def train_with_cross_validation(args, dataset):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_index = 0
    fold_results=[]
    for train_index, val_index in kfold.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader =torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        args.epoch =50#5轮预训练
        args.d_pth = ""
        fold_result_pretrain=train(args, train_loader, val_loader, fold_index)

        #args.d_pth = f"./models/only_dis/pipr/D_best_cross_{fold_index}.pth"
        args.epoch = 50
        args.beta_fake_loss=0.5
        args.beta_real_loss=0.5
        fold_result_finetune = train(args, train_loader, val_loader, fold_index)  # 微调结果

        fold_results.append({
            "fold_index": fold_index,
            "pretrain": fold_result_pretrain,
            "finetune": fold_result_finetune
        })
        fold_index += 1
    avg_results = average_fold_results(fold_results)
    print("Average Results:")
    print_results(avg_results)
    
def average_fold_results(results_list):
    avg_results = {}
    num_folds = len(results_list)
    for key in results_list[0].keys():
        avg_results[key] = sum(result[key] for result in results_list) / num_folds
    return avg_results

def print_results(results):
    for key, value in results.items():
        print(f"{key}: {value:.4f}")        

def train(args,train_loader,val_loader,fold_index):
    '''
    if args.train_dataset == "" and args.test_dataset == "":
        dataset = MyDataset(args.interaction_data, args.sequence_data)
        train_dataset, test_dataset = split_dataset(dataset)
        torch.save(train_dataset, "./data/train_dataset.pth")
        torch.save(test_dataset, "./data/test_dataset.pth")
    else:
        train_dataset = torch.load(args.train_dataset)
        test_dataset = torch.load(args.test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    '''
    D = Dis(args)
    G = Gen(args)
    G.weight_init(mean=0.0, std=0.02)
    if args.d_pth == "":
        D.weight_init(mean=0.0, std=0.02)
        args.is_only_dis = False
    else:
        sd = torch.load(args.d_pth)
        D.load_state_dict(sd)
        args.is_only_dis = False

    G.to(args.device)
    D.to(args.device)
    cond_gan = cgan(G, D).to(args.device)

    # 训练过程
    criterion = nn.BCELoss()
    criterion_gen = nn.BCELoss()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    optimizer_cond_gan = torch.optim.Adam(cond_gan.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8)
    global_step = 0
    best_acc, best_epoch = 0, 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch):
        D.train()
        G.train()
        for i, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.type(torch.float).to(args.device)

            # 训练判别器
            optimizer_D.zero_grad()

            is_gen = torch.zeros((x1.size(0),)).to(args.device)
            real_outputs = D(x1, x2, is_gen)
            # print(real_outputs, y)
            real_loss = criterion(real_outputs, y)
            # only_discriminator
            if args.is_only_dis:
                real_loss.backward()
                optimizer_D.step()
                print(f"Epoch [{epoch + 1}/{args.epoch}], Step [{i + 1}/{len(train_loader)}], "
                      f"D_loss: {real_loss.item():.4f}")
                continue
            true_index = (y == torch.tensor([0, 1], device=args.device)).all(dim=1)
            # print(x1.shape)
            x1 = x1[true_index]
            x2 = x2[true_index]
            y = y[true_index]
            # print(x1.shape)

            fake_labels = np.zeros((x1.size(0), 2), dtype=int)
            fake_labels[:, 0] = 1
            fake_labels = torch.from_numpy(fake_labels).type(torch.float).to(args.device)

            real_labels = np.zeros((x1.size(0), 2), dtype=int)
            real_labels[:, 1] = 1
            real_labels = torch.from_numpy(real_labels).type(torch.float).to(args.device)

            z = np.random.randint(21, size=(x1.size(0), 1500, 15))
            z = torch.from_numpy(z).to(args.device)

            is_gen = torch.ones((x1.size(0),)).to(args.device)
            fake_inputs = G(x1, z)
            fake_outputs = D(x1, fake_inputs.detach(), is_gen)
            fake_loss_a = criterion(fake_outputs, fake_labels)

            fake_inputs = G(x2, z)
            fake_outputs = D(x2, fake_inputs.detach(), is_gen)
            fake_loss_b = criterion(fake_outputs, fake_labels)
            
            d_loss = args.beta_real_loss * real_loss + (fake_loss_a + fake_loss_b) * args.beta_fake_loss
            d_loss.backward()
            # for name, param in D.named_parameters():
            #     print(name, param.grad)
            optimizer_D.step()
            global_step += 1
            # 训练生成器
            for k in range(3):
              optimizer_cond_gan.zero_grad()
              fake_outputs_a = cond_gan(x1, args)
              fake_outputs_b = cond_gan(x2, args)
              # fake_outputs = D(x1, fake_inputs, is_gen)
              g_loss = (criterion_gen(fake_outputs_a, real_labels) + criterion(fake_outputs_b, real_labels)) * 0.5
              # 固定参数
              for name, param in cond_gan.dis.named_parameters():
                  param.requires_grad = False

              for name, param in cond_gan.gen.named_parameters():
                  param.requires_grad = True
              g_loss.backward()

              # 打印各层的梯度
              # for name, param in cond_gan.named_parameters():
              #     if "gen." in name:
              #         print(name, param.grad)

              optimizer_cond_gan.step()
              for param in cond_gan.dis.parameters():
                  param.requires_grad = True
              print(f"Epoch [{epoch + 1}/{args.epoch}], Step [{i + 1}/{len(train_loader)}], "
                  f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")

        # 验证模型
        D.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for x1, x2, y in val_loader:
                x1 = x1.to(args.device)
                x2 = x2.to(args.device)
                y = y.to(args.device)
                
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
            tn, fp, fn, tp = cm.ravel()


            # 计算性能指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            specificity = tn / (tn + fp)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
                    
            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch + 1
                if args.is_only_dis:
                    torch.save(D.state_dict(), os.path.join('./models/only_dis/yeast/', f'D_best_cross_{fold_index}.pth'))
                else:
                    torch.save(D.state_dict(),
                               os.path.join('./models/dis_after_gen/yeast/', f'D_best_{args.beta_real_loss}_{fold_index}.pth'))
                    torch.save(G.state_dict(),
                               os.path.join('./models/dis_after_gen/yeast/', f'G_best_{args.beta_fake_loss}_{fold_index}.pth'))
                print("Best model saved!")
            print("best epoch:",best_epoch)
            print("best acc:",best_acc)
            
            print("混淆矩阵:")
            print(cm)
            print("准确率:", accuracy)
            print("精确率:", precision)
            print("特异性:", specificity)
            print("召回率:", recall)
            print("F1值:", f1)
            print("MCC:", mcc)
            print(f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}")
            print("===============================================")
            if args.is_only_dis:
                with open(f"./log/yeast/only_dis.txt", "a+") as f:
                    f.write(
                        f"Epoch [{epoch + 1}/{args.epoch}] \ncm:{cm} \nAccuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, MCC:{mcc}\n")
                    f.write(f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}\n")
                    f.write("===============================================\n")
                continue
            with open(f"./log/yeast/log_{args.beta_real_loss}_{args.beta_fake_loss}_deeptrio_1006.txt", "a+") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{args.epoch}] \ncm:{cm} \nAccuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, MCC:{mcc}\n")
                f.write(f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}\n")
                f.write("===============================================\n")
    return {
            "cm": cm,
            "accuracy": accuracy,
            "precision": precision,
            "specificity": specificity,
            "recall": recall,
            "f1": f1,
            "mcc": mcc
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interaction_data',default="/2014110079/PPIGAN/PPIGAN/Dataset/yeast core dataset from PIPR/protein.actions.tsv",type=str)
    parser.add_argument('--sequence_data',default="/2014110079/PPIGAN/PPIGAN/Dataset/yeast core dataset from PIPR/protein.dictionary.tsv", type=str)
    parser.add_argument('--epoch', type=int, default=50, help='the maximum number of epochs')
    parser.add_argument('--train_dataset', default="", required=False, type=str,
                        help="The path of the train dataset")
    parser.add_argument('--test_dataset', default="", required=False, type=str,
                        help="The path of the test dataset")
    parser.add_argument('--d_pth', default="", required=False, type=str,
                        help="The path of the discriminator model")
    parser.add_argument('--cuda', default=True, action='store_true', help='Whether apply GPU to train the model')
    parser.add_argument('--beta_real_loss', default=0.5, type=float, help='The weight of the real loss')
    parser.add_argument('--beta_fake_loss', default=0.5, type=float, help='The weight of the fake loss')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size of the training process')
    parser.add_argument('--conv_num', default=10, type=int, help='The number of the convolution layers')
    parser.add_argument('--em_dim', default=15, type=int, help='The dimension of the embedding layer')
    parser.add_argument('--sp_drop', default=0.005, type=float, help='The dropout rate of the spatial dropout layer')
    parser.add_argument('--kernel_rate_1', default=0.16, type=float,
                        help='The kernel rate of the first convolution layer')
    parser.add_argument('--strides_rate_1', default=0.15, type=float,
                        help='The strides rate of the first convolution layer')
    parser.add_argument('--kernel_rate_2', default=0.14, type=float,
                        help='The kernel rate of the second convolution layer')
    parser.add_argument('--strides_rate_2', default=0.25, type=float,
                        help='The strides rate of the second convolution layer')
    parser.add_argument('--filter_num_1', default=150, type=int,
                        help='The number of the filters of the first convolution layer')
    parser.add_argument('--filter_num_2', default=175, type=int,
                        help='The number of the filters of the second convolution layer')
    parser.add_argument('--con_drop', default=0.05, type=float, help='The dropout rate of the convolution layer')
    parser.add_argument('--fn_drop_1', default=0.2, type=float,
                        help='The dropout rate of the first fully connected layer')
    parser.add_argument('--fn_drop_2', default=0.1, type=float,
                        help='The dropout rate of the second fully connected layer')
    parser.add_argument('--node_num', default=256, type=int,
                        help='The number of the nodes of the fully connected layer')

    global args
    args = parser.parse_args()
    if args.cuda:
        args.device = select_device("cuda:0")
    else:
        args.device = select_device("cpu")
    dataset = MyDataset(args.interaction_data, args.sequence_data)
    train_with_cross_validation(args, dataset)