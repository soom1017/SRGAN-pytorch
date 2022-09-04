from torch.utils.data import Dataset
import torchvision.transforms as transforms

import config

import os
from glob import glob
from PIL import Image

def get_path(img_dir):
    img_paths = glob(img_dir + '/*')
    return sorted(img_paths)

def make_lr_img(hr_img: Image):
    lr_size = config.CROP_SIZE // config.SCALE_FACTOR
    lr_img = hr_img.resize((lr_size, lr_size), resample=Image.BICUBIC)

    return lr_img

def load_imgs(img_dir):
    img_paths = get_path(img_dir)
    hr_tensors = []
    lr_tensors = []
    for img_path in img_paths:
        img = Image.open(img_path)

        transform = transforms.RandomCrop(config.CROP_SIZE)
        hr_img = transform(img)
        lr_img = make_lr_img(hr_img)

        transform = transforms.ToTensor()
        hr_tensor = transform(hr_img)
        lr_tensor = transform(lr_img)

        hr_tensors.append(hr_tensor)
        lr_tensors.append(lr_tensor)
    return hr_tensors, lr_tensors

def load_test_img(img_path):
    img = Image.open(img_path)
    test_img_name = img_path.split(os.path.sep)[-1].split('.')[0]

    transform = transforms.ToTensor()
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    return (test_img_name, img_tensor)


class TrainDataset(Dataset):
    def __init__(self, img_dir):
        self.hr_tensors, self.lr_tensors = load_imgs(img_dir)   

    def __len__(self):
        return len(self.hr_tensors)

    def __getitem__(self, index):
        hr_tensor = self.hr_tensors[index]
        lr_tensor = self.lr_tensors[index]

        return (
            lr_tensor,
            hr_tensor
        )