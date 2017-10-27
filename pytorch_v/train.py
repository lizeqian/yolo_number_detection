from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class solver:
    
    def image_read(self, dir):
        normalize = transforms.Normalize(
           mean=[0.485],
           std=[0.229]
        )
        preprocess = transforms.Compose([
           transforms.Scale(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize
        ])
    
        img_pil = Image.open(dir)
        img_tensor = preprocess(img_pil)
        img_tensor.unsqueeze_(0)