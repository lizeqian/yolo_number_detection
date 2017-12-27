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
from lstm import LSTMLayer
import torch.optim as optim
from logger import Logger
from loss import Loss


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
        self.net = net
        
        

class Rand_num(Dataset):
    def __init__(self, csv_path, img_path, img_size, transform=None):
        csv_path = csv_path
        self.img_paths = os.listdir(img_path)
        self.parent_dir = img_path
        self.img_size = img_size
        image_labels = np.genfromtxt(csv_path, delimiter=',')
        image_labels.flatten()
        image_labels = np.reshape(image_labels, [-1, 14, 14, 18])

        self.labels=image_labels

    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        label = self.labels[index]
        
        image_addr = self.parent_dir+'/'+self.img_paths[index]
        images = cv2.imread(image_addr,0)
        img = np.zeros((14, 1, 112, 112), dtype=np.uint8)
        img[0, 0] = cv2.resize(images,(112, 112), interpolation = cv2.INTER_CUBIC)
        size = self.img_size//2
        for y in range(2):
            for x in range(2):
                img[1+x+y*2, 0] = cv2.resize(images[y*size:y*size+size, x*size:x*size+size] ,(112, 112), interpolation = cv2.INTER_CUBIC)
                
        size = self.img_size//3
        for y in range(3):
            for x in range(3):
                img[5+x+y*3, 0] = cv2.resize(images[y*size:y*size+size, x*size:x*size+size] ,(112, 112), interpolation = cv2.INTER_CUBIC)

        return img, label

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return len(self.img_paths)

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/trained.bin'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 2
    load_checkpoint= True
    num_class = 13
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = '../data/detection_test_t.csv'
    img_path = '../data/detection_test_t'
    dataset = Rand_num(csv_path, img_path, 112*4, None)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=2)
    
    

#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    print (images)
#    images=tensor_to_img(images)
#    print (labels)
#    print (images)
    
    net = Net(14*batch_size)
    lstm = LSTMLayer(7*7*(16+5*2), 64, 14*14*(num_class+5*2), 2, batch_size)
    lossfunction = Loss(batch_size)
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': lstm.parameters(), 'lr': 0.0001}], lr=0, weight_decay=0)
    if load_checkpoint:
        net.load_state_dict(torch.load(SAVE_PATH))
        
    net.cuda()
        
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(2000): 
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float()/256, labels.float()
    
            # wrap them in Variable
            inputs = inputs.contiguous().view(-1, 1, 112, 112)            
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # zero the parameter gradients
            optimizer.zero_grad()
            lstm.hidden = lstm.init_hidden()
            lstm.train()
            outputs = net.forward(inputs, False)
            outputs = outputs.contiguous().view(batch_size, 14, -1)
#            print(outputs)
            outputs = lstm.forward(outputs)
#            print(outputs)
            loss, _ = lossfunction.loss_function_vec(outputs, labels, 0.2)
            loss.backward()
            optimizer.step()
            # print statistics
            if epoch % 1 == 0 and i == 0:  
#                outputs = net.forward(inputs, False)
#                loss, accu = net.loss_function_vec(outputs, labels, 0.5, cal_accuracy=True)
                print (datetime.datetime.now())
                print ('Epoch %g'%(epoch))
                print(loss.data.cpu().numpy())

#                logger.scalar_summary('loss', loss.data.cpu().numpy(), epoch)
#                logger.scalar_summary('Accuracy detection TP', accu[0].data.cpu().numpy(), epoch)
#                logger.scalar_summary('Accuracy detection FP', accu[1].data.cpu().numpy(), epoch)
#                logger.scalar_summary('Accuracy IOU', accu[2].data.cpu().numpy(), epoch)
#
#            if epoch % 1 == 0 and i==0:
#                torch.save(net.state_dict(), SAVE_PATH)
#    torch.save(net.state_dict(), SAVE_PATH)
#    print('Finished Training')

