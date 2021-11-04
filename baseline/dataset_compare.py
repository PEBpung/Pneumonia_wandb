import albumentations as A
from albumentations.pytorch import transforms as Atrans

from torchvision import transforms

import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

class GetAlbData(object):
    def __init__(self, data_dir, img_size, bit, data_type=None, mode=None, num_cls=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.mode = mode
        self.num_cls = num_cls
        self.imgs = []

        # 전처리 정의
        if self.mode == 'train':
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=0.5, std=0.5),
                A.HorizontalFlip(),
                Atrans.ToTensorV2()
            ])
        
        elif self.mode == 'val':
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=0.5, std=0.5),
                Atrans.ToTensorV2()
            ])
        
        self.lst_data = os.listdir(self.data_dir)
        len_data = len(self.lst_data)

        for i in range(0, len_data):
            lst_data_file_name = list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))
            self.imgs.extend([(self.lst_data[i], j) for j in lst_data_file_name])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.imgs[idx][0], self.imgs[idx][1])

        # 이미지를 불러오는 부분 
        # 데이터를 그때그때 불러오게 된다.
        img_data = Image.open(img_path)
        if img_data.mode == 'RGB':
            img_data = img_data.convert('L')

        np_img = np.array(img_data)

        for i in range(0, self.num_cls):
            if self.imgs[idx][0] == self.lst_data[i]:
                label = i 

        img = self.transforms(image=np_img)["image"]

        return img, label

class InitTransbData(object):
    def __init__(self, data_dir, img_size, bit, data_type=None, mode=None, num_cls=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_cls = num_cls
        self.mode = mode
        self.imgs = []
        self.label = []

        # 전처리 코드
        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        
        elif self.mode == 'val':
            self.transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        self.lst_data = os.listdir(self.data_dir)
        len_data = len(self.lst_data)

        # 이미지를 불러오는 부분 
        # 데이터를 한번에 불러오고 list에 저장해둔다.
        for i in range(0, len_data):
            lst_data_file_name = list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))
            for img_name in lst_data_file_name:
                lst_data_file_name = os.path.join(self.data_dir, self.lst_data[i], img_name)
                self.imgs.extend([self.transforms(self.pre_transforms(Image.open(lst_data_file_name)))])
                self.label.extend([i])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.label[idx]

        return img, label

    def pre_transforms(self, img_data):
        self.img_data = img_data

        if self.img_data.mode == 'RGB':
            self.img_data = self.img_data.convert('L')

        return  self.img_data


class InitAlbData(object):
    def __init__(self, data_dir, img_size, bit, data_type=None, mode=None, num_cls=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.mode = mode
        self.num_cls = num_cls
        self.imgs = []
        self.label = []

        # 전처리 정의
        if self.mode == 'train':
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=0.5, std=0.5),
                A.HorizontalFlip(),
                Atrans.ToTensorV2()
            ])
        
        elif self.mode == 'val':
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=0.5, std=0.5),
                Atrans.ToTensorV2()
            ])
        
        self.lst_data = os.listdir(self.data_dir)
        len_data = len(self.lst_data)
        
        # 이미지를 불러오는 부분 
        # 데이터를 한번에 불러오고 list에 저장해둔다.
        for i in range(0, len_data):
            lst_data_file_name = list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))
            for img_name in lst_data_file_name:
                lst_data_file_name = os.path.join(self.data_dir, self.lst_data[i], img_name)
                self.imgs.extend([self.pre_transforms(Image.open(lst_data_file_name))])
                self.label.extend([i])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.transforms(image = self.imgs[idx])["image"]
        label = self.label[idx]

        return img, label

    def pre_transforms(self, img_data):
        self.img_data = img_data

        if self.img_data.mode == 'RGB':
            self.img_data = self.img_data.convert('L')

        self.np_img = np.array(self.img_data)
        return  self.np_img


# For test
if __name__ == '__main__':
    train_data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512", "train")
    num_classes =  len(os.listdir(os.path.join(train_data_dir)))
    train_dataset = GetAlbData(train_data_dir, 512, 8, 'img', 'train', num_classes)
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0)

    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)