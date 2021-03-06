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
        img_paths = img_path+'/*.jpg'
        image_addrs = glob.glob(img_paths)
        image_labels = np.genfromtxt(csv_path, delimiter=',')
        image_labels.flatten()
        image_labels = np.reshape(image_labels, [-1, 7, 7, 18])
        N = len(image_addrs)
#        assert N==np.shape(image_labels)[0]
        images = np.zeros((N, 1, img_size, img_size), dtype=np.uint8)

        for n in range(N):
            image_addr = img_path+'/'+str(n)+'.jpg'
            images[n] = np.expand_dims(cv2.imread(image_addr,0), 0)
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
#        print ('\tcalling Dataset:__len__')
        return len(self.images)

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp100000.bin'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 1
    load_checkpoint= True
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = '../data/0.csv'
    img_path = '../data'
    dataset = Rand_num(csv_path, img_path, 448, None)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=2)

#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    print (images)
#    images=tensor_to_img(images)
#    print (labels)
#    print (images)
    
#    net = Net(batch_size)
#    if load_checkpoint:
#        net.load_state_dict(torch.load(SAVE_PATH))
#        
#    net.cuda()
#        
#    thld = np.arange(0,1,0.05)
#    accu_tp=[]
#    accu_fp=[]
#    accu_iou=[]
    for epoch in range(1): 
        for i, data in enumerate(loader, 0):
                # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float()/256, labels.float()
#        
#                # wrap them in Variable
#                
#            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#    
#            threshold=0.5
#            outputs = net.forward(inputs, False)
#            loss, accu = net.loss_function_vec(outputs, labels, threshold, cal_accuracy=True)
#    
#                
#    #            print (datetime.datetime.now())
#    #            print ('Epoch %g'%(epoch))
#                print(loss.data.cpu().numpy())
#                print(accu)
#                accu_tp.append(accu[0].data.cpu().numpy()[0])
#                accu_fp.append(accu[1].data.cpu().numpy()[0])
#                accu_iou.append(accu[2].data.cpu().numpy()[0])
#                
#    plt.plot(thld, accu_tp, 'r')
#    plt.plot(thld, accu_fp, 'b')
#    plt.plot(thld, accu_iou, 'g')
#    plt.show()
#            gt_classes = labels[:, :, :, 5:]
            img = (inputs*256).int().numpy()[0,0]
#            predict_boxes = outputs[:, (7*7*16+7*7*2):].contiguous().view(1, 7, 7, 2, 4)
#            predict_confidence = outputs[:, 7*7*16:(7*7*16+7*7*2)].contiguous().view(1, 7, 7, 2)
#            predict_class = outputs[:,:7*7*16].contiguous().view(1, 7, 7, 16)
#            max_confidence = torch.max(predict_confidence, 3, keepdim = True)
#            threshold = 0.2
#            detect_ob = torch.ge(max_confidence[0], threshold).float()
#            font = cv2.FONT_HERSHEY_PLAIN 
#            directory = os.path.dirname('../bounding_boxes_100/')
#            if not os.path.exists(directory):
#                os.makedirs(directory)
#####for test#######
            for y in range(7):
                for x in range(7):
                    if labels[0,y,x,4]==1:
                        xp, yp, w, h = labels[0,y,x,:4]
                        lu = (int((x+xp)*16*4-w*448/2), int((y+yp)*64-h*448/2))
                        rb = (int((x+xp)*64+w*448/2), int((y+yp)*64+h*448/2))
                        cv2.rectangle(img, lu, rb, 200)
#            for y in range(7):
#                for x in range(7):
#                    if detect_ob.data.cpu().numpy()[0, y, x, 0] == 1:
#                        selection = max_confidence[1].data.cpu().numpy()[0,y,x,0]
#                        class_ = np.argmax(predict_class.data.cpu().numpy()[0,y,x])
#                        gt_class = np.argmax(gt_classes.data.cpu().numpy()[0,y,x])
#                        xp, yp, w, h = predict_boxes.data.cpu().numpy()[0,y,x,selection]
#        #                print((xp, yp, w, h))
#        #                print((x,y))
#                        lu = (int((x+xp)*16-w*112/2), int((y+yp)*16-h*112/2))
#                        rb = (int((x+xp)*16+w*112/2), int((y+yp)*16+h*112/2))
#
#                        if class_ == gt_class:
#                            color = 255
#                        else :
#                            color = 100
#                            cv2.putText(img,str(class_),(lu[0],lu[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
#                            cv2.putText(img,str(gt_class),(rb[0],rb[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
#                            print(gt_class)
#                        cv2.rectangle(img, lu, rb, color)

        #                print(lu)
        #                print(rb)


            write_path = '../'+str(i)+'.jpg'
            cv2.imwrite(write_path,img)
    print('Finished Marking')


    


    

