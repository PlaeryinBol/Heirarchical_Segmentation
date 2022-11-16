import glob
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset


class BodyPartDataset(Dataset):
    def __init__(self, img_path, mean, std, test=False, transform=None):
        self.img_path = img_path
        self.samples = glob.glob(self.img_path + '/**/*.jpg', recursive=True)
        self.test = test
        self.transform = transform
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.samples[idx][:-4] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        if not self.test:
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)
        
        mask = torch.from_numpy(mask).long()
            
        return img, mask
    
