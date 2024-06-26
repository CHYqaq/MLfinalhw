import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
import numpy as np
import os
import matplotlib.pyplot as plt
from congestion_dataset import CongestionDataset, load_annotations
from sklearn.model_selection import train_test_split
from models import GPDL
from utils.metrics import ssim, nrms  # 导入ssim和nrms函数
from tqdm import tqdm  # 用于进度条显示
from math import cos, pi
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

# 定义计算指标的函数
def compute_metrics(pred, target):
    # 确保传递的是张量，而不是NumPy数组
    pred_np = pred.cpu()
    target_np = target.cpu()
    
    nrmse = nrms(pred_np, target_np)
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

# 保存检查点的函数
def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

# 定义余弦退火学习率调度器类
class CosineRestartLr(object):
    def __init__(self, base_lr, periods, restart_weights=[1], min_lr=None, min_lr_ratio=None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float, end: float, factor: float, weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds cumulative_periods {cumulative_periods}')

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in optimizer.param_groups]  # type: ignore

# 使用新的训练方法
def train_model(model, train_loader, epochs=10, learning_rate=0.001, save_path='checkpoints', print_freq=100, save_freq=10000):
    model.train()
    criterion = torch.nn.MSELoss()  # 假设我们使用均方误差损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cosine_lr = CosineRestartLr([learning_rate], [epochs * len(train_loader)], [1], 1e-7)  # 使用余弦退火学习率
    cosine_lr.set_init_lr(optimizer)

    train_loss = []
    iter_num = 0
    all_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as bar:
            for features,labels,_ in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                iter_num += 1
                bar.update(1)
                
                if iter_num % print_freq == 0:
                    print(f"Iteration {iter_num}, Loss: {loss.item():.4f}")

                if iter_num % save_freq == 0:
                    checkpoint(model, iter_num, save_path)
                
                all_loss.append(loss.item())

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    return train_loss, all_loss

# 训练数据量的不同大小
ratio_sizes = [0.8,0.6,0.4,0.2]
train_infos, test_infos = train_test_split(data_infos, test_size=0.2, random_state=42)

# 用于存储不同 ratio 下的结果
nrmse_results = []
ssim_results = []
train_loss_results = []
all_loss_results = []

# 循环不同的训练数据量
for ratio in ratio_sizes:
    print(f"Training with dataset size ratio: {ratio}")
    # 初始化模型
    model = GPDL()
    model.to(device)  # 将模型加载到指定设备（CPU 或 GPU）

    # 创建数据集和数据加载器
    train_loader, test_loader = create_datasets(train_infos, test_infos, pipeline, ratio)
    
    # 训练模型
    train_loss, all_loss = train_model(model, train_loader, epochs=10, learning_rate=0.001, save_path='checkpoints', print_freq=100, save_freq=10000)
    train_loss_results.append(train_loss)
    all_loss_results.extend(all_loss)
    
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
