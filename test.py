import torch
import torchvision.transforms as transforms
import numpy as np

from model import Generator, Discriminator
from data import get_path, load_test_img
import config


def temp():
    gen = Generator()
    disc = Discriminator()

    low_resolution = 24     # 96x96 -> 24x24
    with torch.cuda.amp.autocast():
        for i in range(100):
            x = torch.randn((5, 3, low_resolution, low_resolution))
            gen_out = gen(x)
            disc_out = disc(gen_out)

            print(gen_out.shape)    # (5, 3, 96, 96)
            print(disc_out.shape)   # (5, 1)


def test():
    gen = Generator()
    gen.load_state_dict(torch.load('model_G_epoch_ ... .pth', map_location=torch.device('cpu')))

    gen.eval()
    img_paths = get_path(config.TEST_IMG_DIR)
    for img_path in img_paths:
        test_img_name, test_img = load_test_img(img_path)
        result = gen(test_img)

        result = result.squeeze()
        print(result)
        result = torch.clip(result, min=0, max=1)

        transform = transforms.ToPILImage()
        result_img = transform(result)

        result_img.save(f'outputs_ ... /{test_img_name}_SR.png')
        

if __name__ == "__main__":
    test()

