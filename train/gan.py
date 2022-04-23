import argparse
import os
import time
import numpy as np
import tqdm
import torch
from torch import nn, Tensor
from torch.utils import data
from torchvision.utils import save_image
from torchvision import datasets
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()
    DEFAULT_DATAROOT = os.path.expandvars('~')+'/datasets'
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATAROOT)
    parser.add_argument('-d', '--device', type=str, default='cuda:1')
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-e', '--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_name', type=str, default='gan')
    parser.add_argument('--save_interval', type=int, default=20)
    args = parser.parse_args()
    return args


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim=100):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *self.make_hiddenLayer(latent_dim, 128, normalize=False),
            *self.make_hiddenLayer(128, 256),
            *self.make_hiddenLayer(256, 512),
            *self.make_hiddenLayer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def make_hiddenLayer(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, in_feat=28**2):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


def main():
    args = get_args()
    print(type(args))
    print(chr(128640), args)

    DATAROOT: str = args.data_root
    DEVICE: str = args.device
    BATCH_SIZE: int = args.batch_size
    EPOCHS: int = args.epochs
    LR: float = args.lr
    NUM_WORKERS: int = args.num_workers

    
    SAVE_NAME: str = args.save_name
    SAVE_DIR: str = os.path.join('temp',SAVE_NAME)
    SAVE_INTERVAL: int = args.save_interval
    LOG_FILE = '%s/train.txt' % (SAVE_DIR )
    MODEL_SAVE_PATH = '%s/%s.pth' % (SAVE_DIR,  SAVE_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    open(LOG_FILE, 'w').close()
    open(MODEL_SAVE_PATH, 'w').close()

    # === Load Dataset ===
    train_set = datasets.MNIST(
        root=DATAROOT, download=True, train=True, transform=transforms.ToTensor())
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # === Load model ===
    latent_dim = 100
    generator = Generator(
        img_shape=[1, 28, 28], latent_dim=latent_dim)  # latent_dim: random noise dim
    discriminator = Discriminator(in_feat=28**2)
    adversarial_loss = torch.nn.BCELoss(reduction='sum')

    generator.to(DEVICE)
    discriminator.to(DEVICE)
    adversarial_loss.to(DEVICE)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR)

    # === train ===
    train_start_time=time.time()
    for epoch in range(1, EPOCHS+1):
        print('\033[32m', end='')
        print('[Epoch %d/%d]' % (epoch, EPOCHS), end=' ')
        current_G_lr = optimizer_G.state_dict()['param_groups'][0]['lr']
        current_D_lr = optimizer_D.state_dict()['param_groups'][0]['lr']
        print('G_LR:%f D_LR:%f' % (current_G_lr, current_D_lr), end=' ')
        print('%s' % (DEVICE), end=' ')
        print('Batch:%d' % (BATCH_SIZE), end=' ')
        print('\033[0m')

        # train
        total_G_loss=0
        total_real_loss=0
        total_fake_loss=0
        total_D_loss=0
        total_images=0
        pbar = tqdm.tqdm(train_loader)
        epoch_start_time=time.time()
        for i,(real_images, _) in enumerate(pbar):
            real_images: Tensor = real_images.to(DEVICE)
            batch_images=real_images.size(0)
            total_images+=batch_images

            # set Ground Truth
            valid = torch.ones([real_images.shape[0],1], requires_grad=False).to(DEVICE)
            fake = torch.zeros([real_images.shape[0],1], requires_grad=False).to(DEVICE)
            # pbar.set_description(valid)

            # --------------------------
            #   Train Generator
            # --------------------------
            np_noise = np.random.normal(
                loc=0, scale=1, size=[real_images.shape[0], latent_dim])
            z = torch.from_numpy(np.float32(np_noise))  # z~ N(0,1)
            gen_images: Tensor = generator(z.to(DEVICE))  # 生成数据
            G_loss: Tensor = adversarial_loss(discriminator(gen_images), valid)
            total_G_loss+=G_loss
            G_loss.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            # --------------------------
            #   Train Discriminator
            # --------------------------
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(
                discriminator(gen_images.detach()), fake)
            D_loss: Tensor = (real_loss+fake_loss)/2
            total_real_loss+=real_loss
            total_fake_loss+=fake_loss
            total_D_loss+=D_loss
            D_loss.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()
            pbar.set_description('GLoss:%.6f DLoss:%.6f (realLoss:%.6f,fakeLoss:%.6f)'%(
                G_loss/batch_images,D_loss/batch_images,real_loss/batch_images,fake_loss/batch_images
            ))
        if epoch%SAVE_INTERVAL==0:
            save_images_dir=os.path.join(SAVE_DIR,'images')
            save_images_file=os.path.join(save_images_dir,'%d.png'%epoch)
            os.makedirs(save_images_dir,exist_ok=True)
            save_image(gen_images[:25],save_images_file,nrow=5)
        epcoh_time=time.time()-epoch_start_time
        train_time=time.time()-train_start_time
        log_info = '[%d/%d] [Gloss:%.6f realloss:%.6f fakeloss:%.6f Dloss:%.6f] [time: %.2fmin(%.2fh)/%.2fmin(%.2fh)]' % (
            epoch,EPOCHS,
            G_loss, real_loss, fake_loss, D_loss,
            epcoh_time/60,epcoh_time/60/60,train_time/60,train_time/60/60
        )
        print(log_info)
        with open(LOG_FILE,'a') as f:
            f.write(log_info+'\n')


if __name__ == '__main__':
    main()
