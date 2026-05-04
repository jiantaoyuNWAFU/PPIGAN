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
from sklearn.model_selection import KFold  # 假设你已经导入了正确的 CGAN 类
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
def test(args, test_loader):
    D = Dis(args)  # 初始化判别器
    #G = Gen(args)  # 初始化生成器
    #cond_gan = cgan(G, D).to(args.device)  # 初始化 CGAN
    D.to(args.device)
    # 加载训练好的模型
    D.load_state_dict(torch.load(args.d_pth))
    #G.load_state_dict(torch.load(args.g_pth))

    D.eval()  # 设置为评估模式
    #G.eval()

    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for x1, x2, y in test_loader:
            #print(x1,x2,y)
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)

            is_gen = torch.zeros((x1.size(0),)).to(args.device)
            outputs = D(x1, x2, is_gen)
            #print(outputs)
            outputs = outputs.cpu().numpy()[:, 1]
            #print(outputs)
            y = y.cpu().numpy()[:, 1]
            pred = (outputs > 0.5).astype(int)
            print(pred)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(outputs.tolist())

    # 计算混淆矩阵和性能指标
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("MCC:", mcc)
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Compute precision-recall curve and AP area
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Plot PR curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

    print("AUC:", roc_auc)
    print("AP:", average_precision)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加你的参数解析代码
    # ...
    parser.add_argument('--interaction_data',
                        default="/home/hlw/gengjing/deeptrio-master/data/benchmarks/BioGRID/sa/third_sa_MV_pair.tsv",
                        type=str)
    parser.add_argument('--sequence_data',
                        default="/home/hlw/gengjing/deeptrio-master/data/benchmarks/BioGRID/sa/double_sa_MV_database.tsv",
                        type=str)
    parser.add_argument('--epoch', type=int, default=15, help='the maximum number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='The batch size of the training process')
    parser.add_argument('--cuda', default=True, action='store_true', help='Whether apply GPU to train the model')
    parser.add_argument('--d_pth', default="/home/hlw/gengjing/deeptrio-pytorch1/models/only_dis/human_biogrid/D_best_49_gen(x,z)_deeprio_degree_5.pth", required=False, type=str,
                        help="The path of the discriminator model")
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
        args.device = select_device("cuda:1")
    else:
        args.device = select_device("cpu")

    # 加载测试集数据
    test_dataset = MyDataset(args.interaction_data, args.sequence_data)  # 假设这是你的测试集
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 在测试集上进行测试
    test(args, test_loader)
