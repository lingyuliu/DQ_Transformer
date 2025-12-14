import torch
from data.base_dataset import BaseDataset
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class NullDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

    def __getitem__(self, index):
        return {'A_paths': os.path.join(self.opt.dataroot, '%d.jpg' % index)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.opt.max_dataset_size * self.opt.batch_size


class CelebData(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128, 128],antialias=True)
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l = len(self.file_names)
        print(self.l)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)

        dep_map = Image.open(os.path.join("/home/lly/workspace/mscoco/celeb512/vis_depth", self.file_names[idx][:-4] + "_depth.png")).convert('L')
        dep_map = self.loader(dep_map)

        image_with_dep = torch.cat([image, dep_map], dim=0)
        return image_with_dep

    def __len__(self):
        return self.l
