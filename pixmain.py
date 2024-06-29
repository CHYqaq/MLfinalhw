#epoch:110 maxiteration:20k 
#test_size:0.2
#batch:1
#lr:0.001
import time
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset
from pixmodel import LocalEnhancer, MultiscaleDiscriminator
from data_loader import CreateDataLoader
import math

def lcm(a, b): 
    return abs(a * b) // math.gcd(a, b) if a and b else 0

from options.train_options import TrainOptions

def compute_nrmse(y_true, y_pred):
    result=np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))
    return result/10

def compute_ssim(y_true, y_pred):
    return ssim(y_true, y_pred, data_range=y_pred.max() - y_pred.min())

def main():
    opt = TrainOptions().parse()
    opt.num_workers = 0
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'C:/Users/DELL/Desktop/iter.txt')
    
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print(f'Resuming from epoch {start_epoch} at iteration {epoch_iter}')        
    else:    
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    max_iterations = opt.max_iterations

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    feature_dir = opt.feature_path
    label_dir = opt.label_path
    data_loader = CreateDataLoader(feature_dir, label_dir, batch_size=opt.batchSize, shuffle=not opt.serial_batches, num_workers=opt.num_workers)
    dataset_size = len(data_loader.dataset)
    print(f'#training images = {dataset_size}')

    # 数据集划分
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(data_loader.dataset, train_indices)
    test_dataset = Subset(data_loader.dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_nc = 3
    output_nc = 1
    ngf = 64
    ndf = 64
    n_blocks = 9
    n_local_enhancers = 1
    n_layers_D = 3
    num_D = 3

    generator = LocalEnhancer(input_nc, output_nc, ngf, n_local_enhancers, n_blocks).to(device)
    discriminator = MultiscaleDiscriminator(input_nc + output_nc, ndf, n_layers_D, num_D).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    class GANLoss(nn.Module):
        def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
            super(GANLoss, self).__init__()
            self.real_label = target_real_label
            self.fake_label = target_fake_label
            self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

        def get_target_tensor(self, input, target_is_real):
            target_value = self.real_label if target_is_real else self.fake_label
            if isinstance(input, list):
                return [torch.full_like(i, target_value) for i in input]
            else:
                return torch.full_like(input, target_value)

        def __call__(self, input, target_is_real):
            target_tensor = self.get_target_tensor(input, target_is_real)
            if isinstance(input, list):
                loss = 0
                for i, t in zip(input, target_tensor):
                    loss += self.loss(i, t)
                return loss / len(input)
            else:
                return self.loss(input, target_tensor)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    loss_D_values = []
    loss_G_values = []
    iterations = []

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter_count = 0

        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        with tqdm(total=min(len(train_loader), max_iterations - total_steps), desc=f'Epoch {epoch}') as pbar:
            for i, data in enumerate(train_loader, start=epoch_iter):
                if total_steps >= max_iterations:
                    break

                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                epoch_iter_count += 1

                real_A = data['A'].to(device)
                real_B = data['B'].to(device)
               
                fake_B = generator(real_A)
                
                optimizer_D.zero_grad()
                pred_real = discriminator(torch.cat((real_A, real_B), 1))
                pred_fake = discriminator(torch.cat((real_A, fake_B.detach()), 1))
                loss_D_real = criterionGAN(pred_real, True)
                loss_D_fake = criterionGAN(pred_fake, False)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()
                pred_fake = discriminator(torch.cat((real_A, fake_B), 1))
                loss_G_GAN = criterionGAN(pred_fake, True)
                loss_G_L1 = criterionL1(fake_B, real_B) * 10
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                optimizer_G.step()

                loss_D_values.append(loss_D.item())
                loss_G_values.append(loss_G.item())
                iterations.append(total_steps)

                pbar.update(1)
                pbar.set_postfix({'loss_D': loss_D.item(), 'loss_G': loss_G.item()})

        print(f'Epoch {epoch}/{opt.niter + opt.niter_decay} - Time: {time.time() - epoch_start_time:.3f}s')

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, total_steps {total_steps}')
            torch.save(generator.state_dict(), os.path.join(opt.checkpoints_dir, f'net_G_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(opt.checkpoints_dir, f'net_D_epoch_{epoch}.pth'))
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            generator.update_fixed_params()

        if epoch > opt.niter:
            generator.update_learning_rate()

        if total_steps >= max_iterations:
            print(f'Maximum iterations {max_iterations} reached, stopping training.')
            break

    # 测试模型
    generator.eval()
    nrmse_values = []
    ssim_values = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            fake_B = generator(real_A)
            
            real_B_np = real_B.cpu().numpy().squeeze()
            fake_B_np = fake_B.cpu().numpy().squeeze()
            
            nrmse = compute_nrmse(real_B_np, fake_B_np)
            ssim_val = compute_ssim(real_B_np, fake_B_np)
            
            nrmse_values.append(nrmse)
            ssim_values.append(ssim_val)
            pbar.update(1)
            pbar.set_postfix({'NRMSE': np.mean(nrmse_values), 'SSIM': np.mean(ssim_values)})

    avg_nrmse = np.mean(nrmse_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f'Test NRMSE: {avg_nrmse:.3f}, SSIM: {avg_ssim:.3f}')

    # 绘制损失图像
    plt.figure()
    plt.plot(iterations, loss_D_values, label='Discriminator Loss')
    plt.plot(iterations, loss_G_values, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
