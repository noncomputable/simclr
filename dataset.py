import torch
from torchvision import transforms
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, dataset, out_img_dims):
        super().__init__()

        self.dataset = dataset
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(out_img_dims, scale=(1/3, 1.0), ratio=(1/3, 2.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=.2)],
                p=.8
            )
        ])

    def get_image(self, idx, normalize):
        img, label = self.dataset[idx]

        img = transforms.ToTensor()(img)
        aug_i = self.augment(img)
        aug_j = self.augment(img)
        
        if normalize:
            aug_i = self.normalize(aug_i) 
            aug_j = self.normalize(aug_j)

        pos_pair = (aug_i, aug_j)

        return pos_pair, torch.tensor(0) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.get_image(idx, True)
