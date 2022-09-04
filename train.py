import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from data import TrainDataset
from loss import VGGLoss
from utils import save_checkpoint
import config

from tqdm import tqdm

# models
print("Computation device: ", config.DEVICE)
disc = Discriminator().to(config.DEVICE)
gen = Generator().to(config.DEVICE)

# dataset module
print(f'Train image directory: {config.TRAIN_IMG_DIR}')
train_data = TrainDataset(img_dir=config.TRAIN_IMG_DIR)
train_loader = DataLoader(train_data)
print(">>> Successfully loaded imgs")

# optimizer, loss function
opt_init = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE)
opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(opt_disc, step_size=int(1e5), gamma=0.1)
bce1 = nn.BCELoss()
bce2 = nn.BCELoss()
mse = nn.MSELoss()
vgg_loss = VGGLoss()

## initialize learning (G)
def trainforinit():
    gen.train()
    for epoch in range(config.NUM_EPOCHS_INIT):
        loop = tqdm(train_loader)
        epoch_loss_init = 0.0
        for idx, (img, target) in enumerate(loop):
            img = img.to(config.DEVICE)
            target = target.to(config.DEVICE)

            fake = gen(img)
            loss_init = mse(fake, target)
            # update G
            opt_init.zero_grad()
            loss_init.backward()
            opt_init.step()

            epoch_loss_init += loss_init.item()
        epoch_loss_init /= len(train_loader.dataset)
        print(f'Epoch: {epoch + 1}/{config.NUM_EPOCHS_INIT} mse: {epoch_loss_init:.3f}')

        if (epoch + 1) % 10 == 0:
            save_checkpoint(gen, opt_init, f'model_init_epoch_{epoch + 1}.pth.tar')

def get_g_loss(fake, target):
    disc_fake = disc(fake)
    mse_loss = mse(fake, target)
    content_loss = 2e-6 * vgg_loss(fake, target)
    # min log(1 - D(G(z))) <-> max log(D(G(z)))
    adversarial_loss = 1e-3 * bce1(disc_fake, torch.ones_like(disc_fake))
    g_loss = mse_loss + content_loss + adversarial_loss
    return g_loss

def get_d_loss(fake, target):
    disc_fake = disc(fake.detach())
    disc_real = disc(target)
    # max log(1 - D(G(z)))
    d_loss = bce2(disc_real, torch.ones_like(disc_real)) + bce2(disc_fake, torch.zeros_like(disc_fake))
    return d_loss


## adversarial learning (G, D)           
def train():
    gen.train()
    disc.train()
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader)
        epoch_loss_gen = 0.0
        epoch_loss_disc = 0.0

        for idx, (img, target) in enumerate(loop):
            img = img.to(config.DEVICE)
            target = target.to(config.DEVICE)
            fake = gen(img)
            # update G
            loss_gen = get_g_loss(fake, target)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            # update D
            loss_disc = get_d_loss(fake, target)
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            epoch_loss_gen += loss_gen
            epoch_loss_disc += loss_disc

        epoch_loss_gen /= int(len(train_data)/train_loader.batch_size)
        epoch_loss_disc /= int(len(train_data)/train_loader.batch_size)
        print(f'Epoch: {epoch + 1}/{config.NUM_EPOCHS} g_Loss: {epoch_loss_gen:.3f}')
        print(f'Epoch: {epoch + 1}/{config.NUM_EPOCHS} d_Loss: {epoch_loss_disc:.3f}')

        if (epoch + 1) % config.SAVE_CHECKPOINT_EPOCH == 0:
                save_checkpoint(gen, opt_gen, f'model_G_epoch_{epoch + 1}.pth.tar')
                save_checkpoint(disc, opt_disc, f'model_D_epoch_{epoch + 1}.pth.tar')


if __name__ == "__main__":
    trainforinit()
    train()
