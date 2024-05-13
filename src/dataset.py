from torch.utils.data import Dataset, TensorDataset
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn

trans_pil_24_24 = transforms.Compose(
    [transforms.Resize((24, 24)), transforms.ToTensor()]
)

trans_pil_320_320 = transforms.Compose(
    [transforms.Resize((320, 320)), transforms.ToTensor()]
)

trans_pil = transforms.Compose([transforms.ToTensor()])


def trans_cv_gray(img: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        img (np.ndarray): _description_

    Returns:
        np.ndarray: gray; (c,h,w); c=1
    """
    cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_img = torch.from_numpy(cv_img).float() / 255
    cv_img = cv_img.unsqueeze(0)
    return cv_img


class imgDataset(Dataset):
    def __init__(self, txt_path, transform=trans_pil_24_24):

        self.transform = transform
        self.labels = []
        self.imgs = []

        with open(txt_path, "r") as f:
            self.txt = f.readlines()
        for l in self.txt:
            l = l[:-1]
            l = l.split(" ")
            img = Image.open(l[0])
            img = self.transform(img)
            self.imgs.append(img)
            self.labels.append(int(l[1]))

        self.imgs = torch.stack(self.imgs, dim=0)
        self.labels = torch.tensor(self.labels)
        self.weights = torch.ones(len(self.imgs))

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        weight = self.weights[index]
        return img, label, weight

    def __len__(self):
        return len(self.imgs)
