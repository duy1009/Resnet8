import glob
import numpy as np
from PIL import Image
import random
import os.path as osp

import torch
from utils import processListPath
import torch.optim
import torch.utils.data as data

from torchvision import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(1234) # để sinh ra số random giống nhau

np.random.seed(1234)
random.seed(1234)

# torch.backends.cudnn.deterministic = True # Dùng để giữ kết quả khi training trên GPU
# torch.backends.cudnn.benchmark = False

class ImageTransform():
    def __init__(self, resize):
        self.data_trans = {
            'train': transforms.Compose([
                # data agumentation
                transforms.Resize(resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase = 'train'):
        return self.data_trans[phase](img)

def to_onehot(num, length):
    resl = torch.zeros(length)
    resl[num] = 1
    return resl

def make_data_path_list(root ="./data", phase = "train"):
    target_path_im = osp.join(root+"/"+phase +"/**/*")
    target_path_im = processListPath(glob.glob(target_path_im))

    return target_path_im

class MyDataset(data.Dataset):
    def __init__(self, file_list, classes, transform = None, phase = "train"):
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = classes
        
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # print(self.cnt)
        # self.cnt+=1
        img_path = self.file_list[index] 
        label = self.classes[img_path.split("/")[-2]]

        img = Image.open(img_path).convert('RGB')
        img_trans = self.transform(img, self.phase).to(self.device)
        label_oh = to_onehot(label, len(self.classes))

        return img_trans, (label, label_oh)


