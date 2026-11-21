"""Per-image QCFS-style CIFAR augmentation, with resumable epoch RNG seeds."""
import numpy as np
import torch
from torch.utils.data import Dataset

class CIFARAuthorDataset(Dataset):
    def __init__(self,raw,labels,indices,dataset):
        from torchvision import transforms
        from torchvision.transforms import AutoAugmentPolicy
        from src.data import NORMALIZATION
        self.raw,self.labels,self.indices=raw,labels,indices
        self.transform=transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),transforms.ToTensor(),transforms.Normalize(*NORMALIZATION[dataset])])
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i];x=self.transform(self.raw[j].transpose(1,2,0))
        # Author Cutout follows normalization, one square of side 16.
        y,cx=np.random.randint(32),np.random.randint(32)
        x[:,max(0,y-8):min(32,y+8),max(0,cx-8):min(32,cx+8)]=0
        return x,int(self.labels[j])
