import torch
from torchvision.utils import save_image
import config
import os
from PIL import Image


def save_checkpoint(model, optimizer, filename='temp_checkpoint.pth.tar'):
    print('>>> Saving checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    print('>>> Loading checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open('test_images/' + file)
        with torch.no_grad():
            upscaled_img = gen(
                # test_transform defined in config file
            )
        save_image(upscaled_img * 0.5 + 0.5, f'saved/{file}')
    gen.train()
