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
        self.img_paths = os.listdir(img_path)
        self.parent_dir = img_path
        self.img_size = img_size
        image_labels = np.genfromtxt(csv_path, delimiter=',')
        image_labels.flatten()
        image_labels = np.reshape(image_labels, [-1, 7, 7, 21])

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

        return images, img, label

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return len(self.img_paths)

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp100000.bin'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 1
    load_checkpoint= True
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = '../data/detection_test.csv'
    img_path = '../data/detection_test_t'
    dataset = Rand_num(csv_path, img_path, 112*4, None)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=2)

    net = Net(14)
    if load_checkpoint:
        net.load_state_dict(torch.load(SAVE_PATH))
        
    net.cuda()
        
    accu_tp=[]
    accu_fp=[]
    accu_iou=[]
    for epoch in range(1): 
        for num, data in enumerate(loader, 0):
                # get the inputs
            images, inputs, labels = data
            inputs, labels = inputs.float()[0]/256, labels.float()
#        
#                # wrap them in Variable
#                
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net.forward(inputs, False)
            img = (inputs*256).data.cpu().int().numpy()[0,0]
            img=images.float().numpy()[0]
            

            for i in range(14):
                output = outputs[i]
                predict_boxes = output[(7*7*16+7*7*2):].contiguous().view(1, 7, 7, 2, 4)
                predict_confidence = output[7*7*16:(7*7*16+7*7*2)].contiguous().view(1, 7, 7, 2)
                predict_class = output[:7*7*16].contiguous().view(1, 7, 7, 16)
                max_confidence = torch.max(predict_confidence, 3, keepdim = True)
                threshold = 0.2
                detect_ob = torch.ge(max_confidence[0], threshold).float()
                font = cv2.FONT_HERSHEY_PLAIN 
                directory = os.path.dirname('../bounding_boxes_t/')
                if not os.path.exists(directory):
                    os.makedirs(directory)
    
                for y in range(7):
                    for x in range(7):
                        if detect_ob.data.cpu().numpy()[0, y, x, 0] == 1:
                            selection = max_confidence[1].data.cpu().numpy()[0,y,x,0]
                            class_ = np.argmax(predict_class.data.cpu().numpy()[0,y,x])
                            xp, yp, w, h = predict_boxes.data.cpu().numpy()[0,y,x,selection]
                            if i == 0:
                                lu = (int((x+xp)*16*4-w*112*2), int((y+yp)*16*4-h*112*2))
                                rb = (int((x+xp)*16*4+w*112*2), int((y+yp)*16*4+h*112*2))
                            elif i == 1:
                                lu = (int((x+xp)*16*2-w*112), int((y+yp)*16*2-h*112))
                                rb = (int((x+xp)*16*2+w*112), int((y+yp)*16*2+h*112))
                            elif i == 2:
                                lu = (int((x+xp)*16*2-w*112)+224, int((y+yp)*16*2-h*112))
                                rb = (int((x+xp)*16*2+w*112)+224, int((y+yp)*16*2+h*112))
                            elif i == 3:
                                lu = (int((x+xp)*16*2-w*112), int((y+yp)*16*2-h*112)+224)
                                rb = (int((x+xp)*16*2+w*112), int((y+yp)*16*2+h*112)+224)
                            elif i == 4:
                                lu = (int((x+xp)*16*2-w*112)+224, int((y+yp)*16*2-h*112)+224)
                                rb = (int((x+xp)*16*2+w*112)+224, int((y+yp)*16*2+h*112)+224)
                            elif i == 5:
                                lu = (int((x+xp)*21-w*74), int((y+yp)*21-h*74))
                                rb = (int((x+xp)*21+w*74), int((y+yp)*21+h*74))
                            elif i == 6:
                                lu = (int((x+xp)*21-w*74+448/3), int((y+yp)*21-h*74))
                                rb = (int((x+xp)*21+w*74+448/3), int((y+yp)*21+h*74))
                            elif i == 7:
                                lu = (int((x+xp)*21-w*74+448/3*2), int((y+yp)*21-h*74))
                                rb = (int((x+xp)*21+w*74+448/3*2), int((y+yp)*21+h*74))
                            elif i == 8:
                                lu = (int((x+xp)*21-w*74), int((y+yp)*21-h*74+448/3))
                                rb = (int((x+xp)*21+w*74), int((y+yp)*21+h*74+448/3))
                            elif i == 9:
                                lu = (int((x+xp)*21-w*74+448/3), int((y+yp)*21-h*74+448/3))
                                rb = (int((x+xp)*21+w*74+448/3), int((y+yp)*21+h*74+448/3))
                            elif i == 10:
                                lu = (int((x+xp)*21-w*74+448/3*2), int((y+yp)*21-h*74+448/3))
                                rb = (int((x+xp)*21+w*74+448/3*2), int((y+yp)*21+h*74+448/3))
                            elif i == 11:
                                lu = (int((x+xp)*21-w*74), int((y+yp)*21-h*74+448/3*2))
                                rb = (int((x+xp)*21+w*74), int((y+yp)*21+h*74+448/3*2))
                            elif i == 12:
                                lu = (int((x+xp)*21-w*74+448/3), int((y+yp)*21-h*74+448/3*2))
                                rb = (int((x+xp)*21+w*74+448/3), int((y+yp)*21+h*74+448/3*2))
                            elif i ==13:
                                lu = (int((x+xp)*21-w*74+448/3*2), int((y+yp)*21-h*74+448/3*2))
                                rb = (int((x+xp)*21+w*74+448/3*2), int((y+yp)*21+h*74+448/3*2))
                            color = 255
                            cv2.putText(img,str(class_),(lu[0],lu[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
                            cv2.rectangle(img, lu, rb, color)
    
                write_path = '../bounding_boxes_t/'+str(num)+'.jpg'
            cv2.imwrite(write_path,img)


    


    

