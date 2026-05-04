# !/user/bin/env python3
# -*- coding: utf-8 -*-
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

def export_to_onnx(args, test_loader):
    # 初始化判别器
    D = Dis(args).to(args.device)
    D.load_state_dict(torch.load(args.d_pth))
    D.eval()

    # 获取一个测试样本用于确定输入形状
    sample_x1, sample_x2, _ = next(iter(test_loader))
    sample_x1 = sample_x1.to(args.device)
    sample_x2 = sample_x2.to(args.device)

    # 定义输入名称和动态轴(如果需要处理不同batch size)
    input_names = ["protein1", "protein2", "is_gen"]
    output_names = ["output"]
    dynamic_axes = {
        'protein1': {0: 'batch_size'},
        'protein2': {0: 'batch_size'},
        'is_gen': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    # 创建一个虚拟的is_gen输入(与您的测试代码一致)
    dummy_is_gen = torch.zeros((sample_x1.size(0),)).to(args.device)

    # 导出模型为ONNX格式
    torch.onnx.export(
        D,  # 要导出的模型
        (sample_x1, sample_x2, dummy_is_gen),  # 模型输入(元组)
        args.onnx_output_path,  # 输出文件路径
        export_params=True,  # 导出模型参数
        opset_version=13,  # ONNX算子集版本
        do_constant_folding=True,  # 优化常量折叠
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True
    )

    print(f"Model successfully exported to {args.onnx_output_path}")
def validate_onnx_model(args, test_loader):
    import onnxruntime as ort
    
    # 加载ONNX模型
    ort_session = ort.InferenceSession(args.onnx_output_path)
    
    # 获取一个测试batch
    x1, x2, y = next(iter(test_loader))
    print(type(x1))
    print(x2)
    x1 = x1.numpy()
    x2 = x2.numpy()
    is_gen = np.zeros((x1.shape[0],), dtype=np.float32)
    print(ort_session.get_inputs())
    # 运行ONNX推理
    ort_inputs = {
        ort_session.get_inputs()[0].name: x1,
        ort_session.get_inputs()[1].name: x2
    }
    ort_outs = ort_session.run(None, ort_inputs)
    
    # 运行原始PyTorch模型进行比较
    D = Dis(args).to(args.device)
    D.load_state_dict(torch.load(args.d_pth))
    D.eval()
    
    with torch.no_grad():
        torch_out = D(
            torch.from_numpy(x1).to(args.device),
            torch.from_numpy(x2).to(args.device),
            torch.from_numpy(is_gen).to(args.device)
        ).cpu().numpy()
    
    # 比较结果
    np.testing.assert_allclose(
        torch_out, ort_outs[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    print("ONNX model outputs match PyTorch outputs!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... 保留原有参数 ...
    parser.add_argument('--interaction_data',
                        default="/home/hlw/gengjing/deeptrio-master/data/human/test_data0.2.txt",
                        type=str)
    parser.add_argument('--sequence_data',
                        default="/home/hlw/gengjing/deeptrio-master/data/human/SEQ-Supp-ABCD.tsv",
                        type=str)
    parser.add_argument('--epoch', type=int, default=15, help='the maximum number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='The batch size of the training process')
    parser.add_argument('--cuda', default=True, action='store_true', help='Whether apply GPU to train the model')
    parser.add_argument('--d_pth',
                        default="/home/hlw/gengjing/deeptrio-pytorch1/models/dis_after_gen/human-human/D_best_0.8_gen(x,z)_0.2.pth",
                        required=False, type=str,
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

    # 添加ONNX导出参数
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format')
    parser.add_argument('--onnx_output_path',
                        default="discriminator_model.onnx",
                        type=str,
                        help='Path to save ONNX model')

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
    #export_to_onnx(args, test_loader)
    validate_onnx_model(args, test_loader)
