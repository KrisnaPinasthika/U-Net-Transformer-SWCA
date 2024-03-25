import torch
import numpy as np
from torchvision import transforms
import skimage.io as io
from torch.utils.data import Dataset

class DiodeDataLoader(Dataset):
    """Some Information about DiodeDataLoader"""

    def __init__(self, data_frame, max_depth, img_depth_dim, is_test=False):
        super(DiodeDataLoader, self).__init__()
        self.data_frame = data_frame
        self.min_depth = 0.1
        self.max_depth = max_depth
        self.img_depth_dim = img_depth_dim
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(self.img_depth_dim, antialias=None)
        ])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.)
        
        self.is_test = is_test
        

    def __getitem__(self, index):
        image_path, depth_path, mask_path = self.data_frame.iloc[index]
        
        # Todo: read images
        img = self.norm(self.transform(io.imread(image_path)))
        
        # Todo: depth processing
        depth_map = np.load(depth_path).squeeze()
        mask = np.load(mask_path)
        mask = mask > 0

        max_depth = min(self.max_depth, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth) 
        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = self.transform(depth_map)
        
        if not self.is_test:
            img = self.color_channel_swap(img)
            
            condition = np.random.randn() > 0.5
            if condition:
                img = self.horizontal_flip(img)
                depth_map = self.horizontal_flip(depth_map)
        
        depth_map = depth_map / self.max_depth # [0, 1]
        
        return img, depth_map

    def __len__(self):
        return len(self.data_frame)

    def color_channel_swap(self, image):
        # Get the number of channels in the image
        num_channels = image.shape[0]
        # Generate random indices for channel swapping
        indices = torch.randperm(num_channels)
        # Swap the color channels
        swapped_image = image[indices, :, :]

        return swapped_image