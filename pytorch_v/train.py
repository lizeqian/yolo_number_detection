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
from torch.utils.data.sampler import RandomSampler
from model.network import Net
import torch.optim as optim

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
        img_path = img_path+'/*.png'
        image_addrs = glob.glob(img_path)
        image_labels = np.genfromtxt(csv_path, delimiter=',')
        image_labels = np.reshape(image_labels, [-1, 7, 7, 7])
        N = len(image_addrs)
        assert N==np.shape(image_labels)[0]
        images = np.zeros((N, 1, img_size, img_size), dtype=np.uint8)

        for n in range(N):
            images[n] = np.expand_dims(cv2.imread(image_addrs[n],0), 0)
        self.transform = transform
        self.images=images
        self.labels=image_labels
        self.classes=('56', '79')

    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        img = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        print ('\tcalling Dataset:__len__')
        return len(self.images)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    batch_size = 50
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = '../data/detection_training.csv'
    img_path = '../data/detection_training'
    dataset = Rand_num(csv_path, img_path, 112, None)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=2)

#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    print (images)
#    images=tensor_to_img(images)
#    print (labels)
#    print (images)
    
    net = Net(batch_size)
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(200): 
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()
    
            # wrap them in Variable
            
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            #print (inputs)
            outputs = net.forward(inputs, True)
            loss = net.loss_function_vec(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            #running_loss += loss.data[0]
            if i % 5 == 0:    # print every 2000 mini-batches
                print(loss)
    
    print('Finished Training')

