import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import datetime
import cv2
import glob
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from model.network import Net
import torch.optim as optim
from logger import Logger


def tensor_to_img(img, mean=0, std=1):
        img = img.numpy()[0]
        img = (img*std+ mean)
        img = img.astype(np.uint8)
        #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        return img

class Solver:
    def __init__(self, batch_size, epoch_num, net):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.net = net;
        
        

class Rand_num(Dataset):
    def __init__(self, csv_path, img_path, img_size, transform=None):
        csv_path = csv_path
        self.img_paths = img_path
        image_labels = np.genfromtxt(csv_path, delimiter=',')
        image_labels.flatten()
        self.num_classes = 13
        image_labels = np.reshape(image_labels, [-1, 14, 14, self.num_classes+5])

        self.transform = transform
        self.labels=image_labels

    def __getitem__(self, index):
        image_addr = self.img_paths+'/'+str(index)+'.jpg'
        img = np.expand_dims(cv2.imread(image_addr,0), 0)
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return len(self.labels)

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp.bin'
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 50
    load_checkpoint= True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = '../data/test.csv'
    img_path = '../data/test'
    dataset = Rand_num(csv_path, img_path, 224, None)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=1)
    print("data loaded")
#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    print (images)
#    images=tensor_to_img(images)
#    print (labels)
#    print (images)
    
    net = Net(batch_size)
    if load_checkpoint:
        net.load_state_dict(torch.load(SAVE_PATH))
        print("Model loaded")
#    net.cuda()
        
    thld = np.arange(0,1,0.05)
    accu_tp=[]
    accu_fp=[]
    accu_iou=[]

    threshold = 0.2
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.float()/256, labels.float()

        # wrap them in Variable
        
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

#                threshold=0.5
        net.eval()
        outputs = net.forward(inputs)
        loss, accu = net.loss_function_vec(outputs, labels, threshold, cal_accuracy=True)

        
#            print (datetime.datetime.now())
#            print ('Epoch %g'%(epoch))
        print(loss.data.cpu().numpy())
        print(accu)
        accu_tp.append(accu[0].data.cpu().numpy()[0])
        accu_fp.append(accu[1].data.cpu().numpy()[0])
        accu_iou.append(accu[2].data.cpu().numpy()[0])
                



    


    

