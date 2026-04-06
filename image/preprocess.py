from os import path
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import transforms
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import torchvision
import os
from kaggle.api.kaggle_api_extended import KaggleApi
class ImageDataset(Dataset):
    def __init__(self,data_dir,class_mapping,transform=None,maxdata=100,device="cpu"):
        super().__init__()
        self.data_dir = data_dir
        self.class_mapping = class_mapping
        self.transform = transform
        self.maxdata = maxdata
        self.device = device
        self.load_data()
    
    def load_data(self):
        data = []
        for phrase in os.listdir(self.data_dir):
            phrase_path = os.path.join(self.data_dir,phrase)
            if not os.path.isdir(phrase_path): continue
            for label in os.listdir(phrase_path):
                label_path = os.path.join(phrase_path,label)
                if not os.path.isdir(label_path): continue
                num=0
                label_code = self.class_mapping.get(label, -1) 
                for file in os.listdir(label_path):
                    if self.maxdata is not None and num >= self.maxdata: break
                    image_path = os.path.join(label_path,file)
                    data.append((image_path, label_code))
                    num += 1
        self.data = pd.DataFrame(data, columns=["image_path", "label"])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.transform:
            try:
                image = Image.open(self.data.iloc[index,0]).convert("RGB")
            except:
                print("Bad image:", self.data.iloc[index,0])
                return torch.zeros(3, 224, 224), -1, "bad"
            image = self.transform(image)
            return image,self.data.iloc[index,1], self.data.iloc[index,0]
        return self.data.iloc[index,0], self.data.iloc[index,1], self.data.iloc[index,1]
    def save_data(self,save_dir):
        if not os.path.exists(save_dir):
            print("directory not exist")
            return
        Dataset_dic = {
            "data":self.data[0],
            "label":self.data[1],
        }
        torch.save(Dataset_dic,save_dir)
        print("dataset saved in ",save_dir)

class CIFAR10WithOriginal(torchvision.datasets.CIFAR10):
    def __init__(self, *args, samples_per_class=None, **kwargs):
        super().__init__(*args, **kwargs)

        if samples_per_class is not None:
            targets = np.array(self.targets)
            indices = []

            for c in range(10):
                class_idx = np.where(targets == c)[0]
                selected = np.random.choice(class_idx, samples_per_class, replace=False)
                indices.extend(selected)

            self.data = self.data[indices]
            self.targets = targets[indices].tolist()
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        original_img = transforms.ToTensor()(self.data[index])  # raw image
        return img, original_img, label

def load_data(data_dir, class_mapping, transform=None,maxdata=None,save=False,device="cpu"):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    else:
        transform = transform
    data = ImageDataset(data_dir, class_mapping=class_mapping, transform=transform,maxdata=maxdata,device=device)
    train_size = int(0.8 * len(data))
    eval_size = len(data) - train_size
    train_dataset, eval_dataset = random_split(data, [train_size, eval_size])
    if save:   
        data.save_data(os.path.join(data_dir,"Cleaned_data_train.pt"))
    return train_dataset, eval_dataset

def download_dataset(data_dir):
    if  not os.path.exists(os.path.join(data_dir)):
        os.makedirs(data_dir, exist_ok=True)  
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('marquis03/flower-classification', path=data_dir, unzip=True)
        print("Dataset downloaded and extracted to ", data_dir)
    else:
        print("Dataset already exists at ", data_dir)