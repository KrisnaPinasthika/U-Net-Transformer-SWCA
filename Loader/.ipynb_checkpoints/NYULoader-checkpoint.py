import torch
import numpy as np 
import os
from PIL import Image
from torchvision import transforms
import random
from itertools import permutations

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomChannelSwap(object):
    def __init__(self, p):
        self.p = p
        self.indices = list(permutations(range(3), 3))

    def __call__(self, image):
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if random.random() < self.p:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return image
    
class NYUDepth(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, df, img_size=(240, 320), data_path=None, is_test:bool=False):
        super(NYUDepth, self).__init__()
        self.df = df
        self.img_size = img_size
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(self.img_size, antialias=None), 
        ])
        self.is_test = is_test

    def __getitem__(self, index):
        img_path, depth_path = self.df['image'].iloc[index], self.df['depth'].iloc[index]
        
        if self.data_path is not None: 
            img_path = os.path.join(self.data_path, img_path)
            depth_path = os.path.join(self.data_path, depth_path)
        
        # Todo: read image and depth
        img = Image.open(img_path).crop((43, 45, 608, 472))
        depth = Image.open(depth_path).crop((43, 45, 608, 472))
        
        # Todo: Training case
        if not self.is_test:
            if np.random.randn() > 0.4:
                self.transform.transforms.append(transforms.RandomHorizontalFlip(p=1.))
            
            channel_swap = RandomChannelSwap(p=.5)
            img = channel_swap(img)
            
        img = self.transform(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (img)
        
        if not self.is_test:
            depth = self.transform(depth) * 1000. # satuan jadi cm [10, 1000]
        else: 
            depth = self.transform(np.asarray(depth, dtype=np.float32)) / 1000 # satuan jadi meter [0.1, 10]
            return img, depth 
        
        # Balik nilai depth dan Cliping range
        depth = 1000. / torch.clamp(depth, 10, 1000) # Dibalik supaya bobot jarak yang lebih besar 
        return img, depth
    
    def __len__(self):
        return len(self.df)

if __name__ == '__main__': 
    import pandas as pd 

    df_train = pd.read_csv(r'../dataset/nyu_data/data/nyu2_train.csv', header=None, names=["image", "depth"])
    df_train["scene"] = df_train["image"].apply(lambda x: "_".join(x.split("/")[2].split("_")[:-2]))
    df_test = pd.read_csv(r'../dataset/nyu_data/data/nyu2_test.csv', header=None, names=["image", "depth"])
    
    from torch.utils.data import DataLoader
    loader = DataLoader(
        NYUDepth(
            df=df_test, 
            img_size=(192, 256), 
            data_path=r'../dataset/nyu_data/', 
            is_test=True
        ), 
        batch_size=32, 
        shuffle=False
    )
    
    data, depth = iter(loader).next()
    
    print(data.shape, depth.shape)
    
    idx = 17
    print(depth[idx][0])
    
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 12))
    ax[0].imshow(data[idx].permute(1, 2, 0))
    ax[1].imshow(depth[idx][0])
    
    plt.show()