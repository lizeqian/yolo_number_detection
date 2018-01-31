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
from network2 import Net
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
        self.csv_paths = csv_path
        self.img_paths = img_path
        self.file_count = sum(len(files) for _, _, files in os.walk(img_path))
        self.num_classes = 21
        self.num_cells = 28

        self.transform = transform
        #self.labels=image_labels

    def __getitem__(self, index):
        image_addr = self.img_paths+'/'+str(index)+'.jpg'
        label_addr = self.csv_paths+'/'+str(index)+'.csv'
        img = np.expand_dims(cv2.imread(image_addr,0), 0)
        #label = self.labels[index]

        image_labels = np.genfromtxt(label_addr, delimiter=',')
        image_labels.flatten()
        image_labels = np.reshape(image_labels, [self.num_cells, self.num_cells, self.num_classes+5])


        if self.transform is not None:
            img = self.transform(img)

        return img, image_labels

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return self.file_count


if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp_2.pth'
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 1
    load_checkpoint= True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = 'validation28_label'
    img_path = 'validation28'
    dataset = Rand_num(csv_path, img_path, 448, None)
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

    ave_tp  = []
    ave_fp  = []
    ave_gt  = []
    ave_pt  = []
    ave_iou = []
    ave_cls = []
    threshold = 0.2
    for threshold in thld:
        print("thld is %g"%(threshold))
        accu_tp=[]
        accu_fp=[]
        accu_gt=[]
        accu_pt=[]
        accu_iou=[]
        accu_cls=[]
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float()/256, labels.float()

            # wrap them in Variable

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

#                    threshold=0.5
            net.eval()
            outputs = net.forward(inputs)
            loss, accu = net.loss_function_vec(outputs, labels, threshold, cal_accuracy=True)


#                print (datetime.datetime.now())
#                print ('Epoch %g'%(epoch))
            #print(loss.data.cpu().numpy())
            #print("thld is %g"%(threshold))
            #print(accu)
            #print("%gth img pt is %g"%(i, accu[0][3].data.cpu().numpy()[0]))
            accu_tp.append(accu[0][0].data.cpu().numpy()[0])
            accu_fp.append(accu[0][1].data.cpu().numpy()[0])
            accu_gt.append(accu[0][2].data.cpu().numpy()[0])
            accu_pt.append(accu[0][3].data.cpu().numpy()[0])
            accu_iou.append(accu[1].data.cpu().numpy()[0])
            accu_cls.append(accu[2].data.cpu().numpy()[0])
        ave_tp.append(np.mean(accu_tp))
        ave_fp.append(np.mean(accu_fp))
        ave_gt.append(np.mean(accu_gt))
        ave_pt.append(np.mean(accu_pt))
        ave_iou.append(np.mean(accu_iou))
        ave_cls.append(np.mean(accu_cls))


    print("overall result")
    print(ave_tp)
    print(ave_fp)
    print(ave_gt)
    print(ave_pt)
    print(ave_iou)
    print(ave_cls)









