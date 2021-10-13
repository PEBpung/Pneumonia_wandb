from torch.utils.data import Dataset
import cv2
import torch

class XrayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        label = self.labels[index]
        label = torch.tensor([label])
        
        return image, label