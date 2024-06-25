# main.py
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import numpy as np
from congestion_dataset import CongestionDataset
from models import GPDL
import os

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

# 初始化数据集
test_dataset = CongestionDataset(
    dataroot=dataroot,
    pipeline=pipeline,
    test_mode=True
)

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义NRMSE计算函数
def NRMSE(pred, target):
    mse = mean_squared_error(target, pred)
    rmse = np.sqrt(mse)
    norm = np.sqrt(np.mean(target ** 2))
    return rmse / norm

# 定义计算指标的函数
def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()
    
    nrmse = NRMSE(pred_np, target_np)
    ssim_value = ssim(pred_np, target_np, data_range=target_np.max() - target_np.min())
    
    return nrmse, ssim_value

# 测试模型的函数
def test_model(model, data_loader):
    model.eval()
    nrmse_list = []
    ssim_list = []
    
    with torch.no_grad():
        for features, _, label_path in data_loader:
            features = features.to(device)
            
            outputs = model(features)
            
            # Assuming ground truth labels are not available during test_mode
            # Evaluate the outputs with some placeholder targets if needed
            # Here, I assume that evaluation on actual targets is not needed in test_mode
            # If actual targets are available, you can modify this section
            
            # For demonstration purposes, using features as targets (remove in actual implementation)
            # This is just a placeholder
            dummy_target = features
            for i in range(outputs.size(0)):
                nrmse, ssim_value = compute_metrics(outputs[i], dummy_target[i])
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
    
    for epoch in range(epochs):
        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 初始化模型
model = GPDL()
model.to(device)  # 将模型加载到指定设备（CPU 或 GPU）

# 加载预训练权重路径示例
pretrained_path = 'D:/training_set_forfinal/congestion/pretrained_weights.pth'  # 替换为实际的预训练权重路径

# 检查路径是否存在
if os.path.exists(pretrained_path):
    model.init_weights(pretrained=pretrained_path)
else:
    print(f'预训练权重文件未找到: {pretrained_path}')
    # 如果没有预训练权重，则训练模型并保存权重
    train_dataset = CongestionDataset(dataroot=dataroot, pipeline=pipeline, test_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    train_model(model, train_loader, epochs=10, learning_rate=0.001)
    
    # 保存训练后的模型权重
    torch.save({'state_dict': model.state_dict()}, pretrained_path)
    print(f'模型权重已保存到: {pretrained_path}')

# 测试模型
mean_nrmse, mean_ssim = test_model(model, test_loader)

# 输出结果
print(f'Mean NRMSE: {mean_nrmse:.4f}')
print(f'Mean SSIM: {mean_ssim:.4f}')
