import time
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm  # 引入 tqdm
from pixmodel import LocalEnhancer, MultiscaleDiscriminator
from data_loader import CreateDataLoader
import math

def lcm(a, b): 
    return abs(a * b) // math.gcd(a, b) if a and b else 0

from options.train_options import TrainOptions

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
    max_iterations = opt.max_iterations  # 设置最大迭代次数

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
            if isinstance(input, list):
                return [torch.full_like(t, self.real_label if target_is_real else self.fake_label) for t in input]
            else:
                return torch.full_like(input, self.real_label if target_is_real else self.fake_label)

        def __call__(self, input, target_is_real):
            if isinstance(input, list):
                loss = 0
                for inp in input:
                    target_tensor = self.get_target_tensor(inp, target_is_real)
                    loss += self.loss(inp, target_tensor)
                return loss / len(input)
            else:
                target_tensor = self.get_target_tensor(input, target_is_real)
                return self.loss(input, target_tensor)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    loss_D_values = []
    loss_G_values = []

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_loss_D = 0
        epoch_loss_G = 0
        epoch_iter_count = 0

        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        with tqdm(total=min(len(data_loader), max_iterations - total_steps), desc=f'Epoch {epoch}') as pbar:
            for i, data in enumerate(data_loader, start=epoch_iter):
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

                epoch_loss_D += loss_D.item()
                epoch_loss_G += loss_G.item()

                pbar.update(1)
                pbar.set_postfix({'loss_D': loss_D.item(), 'loss_G': loss_G.item()})

        avg_loss_D = epoch_loss_D / epoch_iter_count
        avg_loss_G = epoch_loss_G / epoch_iter_count
        loss_D_values.append(avg_loss_D)
        loss_G_values.append(avg_loss_G)

        print(f'Epoch {epoch}/{opt.niter + opt.niter_decay} - Loss D: {avg_loss_D:.3f}, Loss G: {avg_loss_G:.3f}, Time: {time.time() - epoch_start_time:.3f}s')

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

    # Plot loss values
    plt.figure()
    plt.plot(range(start_epoch, start_epoch + len(loss_D_values)), loss_D_values, label='Discriminator Loss')
    plt.plot(range(start_epoch, start_epoch + len(loss_G_values)), loss_G_values, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
