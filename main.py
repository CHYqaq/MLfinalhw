import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from congestion_dataset import CongestionDataset, load_annotations
from sklearn.model_selection import train_test_split
from models import GPDL

# 检测是否有 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据转换管道
pipeline = [
    Lambda(lambda x: {'feature': torch.from_numpy(x['feature']).permute(2, 0, 1).float(), 
                      'label': torch.from_numpy(x['label']).permute(2, 0, 1).float() if 'label' in x else None, 
                      'label_path': x['label_path']}),
]

# 数据根目录路径
dataroot = "D:/training_set_forfinal/congestion"  # 替换为实际的数据根目录路径

# 加载数据注释
data_infos = load_annotations(dataroot)

# 定义训练和测试数据集和数据加载器的函数
def create_datasets(train_infos, test_infos, pipeline, ratio):
    train_subset_infos, _ = train_test_split(train_infos, test_size=ratio, random_state=42)
    train_dataset = CongestionDataset(data_infos=train_subset_infos, pipeline=pipeline, test_mode=False)
    test_dataset = CongestionDataset(data_infos=test_infos, pipeline=pipeline, test_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

# 定义SSIM计算函数
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# 定义NRMSE计算函数
def NRMSE(pred, target):
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.mean(target ** 2))
    return rmse / norm

# 定义计算指标的函数
def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()
    
    nrmse = NRMSE(pred_np, target_np)
    ssim_value = ssim(pred_np, target_np)
    
    return nrmse, ssim_value

# 测试模型的函数
def test_model(model, data_loader):
    model.eval()
    nrmse_list = []
    ssim_list = []
    
    with torch.no_grad():
        for features, label, label_path in data_loader:
            features = features.to(device)
            label = label.to(device)
            outputs = model(features)
            
            for i in range(outputs.size(0)):
                nrmse, ssim_value = compute_metrics(outputs[i], label[i])
                nrmse_list.append(nrmse)
                ssim_list.append(ssim_value)
    
    mean_nrmse = np.mean(nrmse_list)
    mean_ssim = np.mean(ssim_list)
    
    return mean_nrmse, mean_ssim

# 训练模型的函数
def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    model.train()
    criterion = torch.nn.MSELoss()  # 假设我们使用均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return train_loss

# 训练数据量的不同大小
ratio_sizes = [0.8, 0.6, 0.4, 0.2]
train_infos, test_infos = train_test_split(data_infos, test_size=0.2, random_state=42)

# 用于存储不同 ratio 下的结果
nrmse_results = []
ssim_results = []
train_loss_results = []

# 循环不同的训练数据量
for ratio in ratio_sizes:
    print(f"Training with dataset size ratio: {ratio}")
    # 初始化模型
    model = GPDL()
    model.to(device)  # 将模型加载到指定设备（CPU 或 GPU）

    # 创建数据集和数据加载器
    train_loader, test_loader = create_datasets(train_infos, test_infos, pipeline, ratio)
    
    # 训练模型
    train_loss = train_model(model, train_loader, epochs=10, learning_rate=0.001)
    train_loss_results.append(train_loss)
    
    # 测试模型
    mean_nrmse, mean_ssim = test_model(model, test_loader)
    nrmse_results.append(mean_nrmse)
    ssim_results.append(mean_ssim)
    
    # 输出结果
    print(f"Ratio: {ratio}")
    print(f"Mean NRMSE: {mean_nrmse:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    
    # 画出训练损失随迭代次数的变化曲线
    plt.plot(range(1, len(train_loss)+1), train_loss, label=f'Ratio: {ratio}')

# 显示所有训练损失曲线
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs for Different Ratios')
plt.legend()
plt.show()

# 绘制不同 ratio 对应的 NRMSE 和 SSIM 曲线
plt.figure()
plt.plot(ratio_sizes, nrmse_results, marker='o', label='NRMSE')
plt.plot(ratio_sizes, ssim_results, marker='o', label='SSIM')
plt.xlabel('Train Size Ratio')
plt.ylabel('Metric Value')
plt.title('NRMSE and SSIM for Different Train Size Ratios')
plt.legend()
plt.show()
