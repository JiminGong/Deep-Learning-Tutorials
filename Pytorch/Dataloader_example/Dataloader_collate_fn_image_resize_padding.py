import os
import math
import torch
import glob
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(__file__)

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, transforms = None):
        self.image_list = image_list
        self.nSamples = len(image_list)
        self.transforms = transforms

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        img = Image.open(img).convert('RGB')
        if self.transforms is not None :
            img = self.transforms(img)
        label = np.random.randint(0,5, (1)) # 임의의 랜덤 레이블
        return img, label

class Image_Pad(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class My_Collate(object):
    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        resized_max_w = self.imgW
        transform = Image_Pad((3, self.imgH, resized_max_w))

        resized_images = []
        for idx, image in enumerate(images):
            print(f'{idx} 번째 데이터 shape :', np.array(image).shape)
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)

            transformed_image = transform(resized_image)
            resized_images.append(transformed_image)

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        return image_tensors, labels



if __name__ == '__main__':
    imgH = 64
    imgW = 256

    image_list = glob.glob(os.path.join(os.path.dirname(__file__), '*.jpg'))
    image_list = sorted(image_list)
    
    transform = transforms.Compose([
            transforms.Resize((64,256)),
            transforms.ToTensor(),
        ])
    My_collate = My_Collate(imgH=imgH, imgW=imgW)
    
    dataset_with_transform = ListDataset(image_list, transforms = transform)
    dataset = ListDataset(image_list)

    data_loader_with_transform = torch.utils.data.DataLoader(dataset_with_transform, batch_size=len(image_list), shuffle=False)
    data_loader_with_collate_fn = torch.utils.data.DataLoader(dataset, batch_size=len(image_list), shuffle=False, collate_fn=My_collate)
    
    data_with_transform = next(iter(data_loader_with_transform))
    data_with_collate_fn = next(iter(data_loader_with_collate_fn))

    data = torch.vstack([data_with_transform[0], data_with_collate_fn[0]])

    grid = torchvision.utils.make_grid(data, nrow = len(image_list))
    plt.imshow(grid.permute(1,2,0))
    plt.show()
