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
        image_labels = np.reshape(image_labels, [-1, 7, 7, 21])
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
    

    img = cv2.imread("test1.png", 0)

#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    print (images)
#    images=tensor_to_img(images)
#    print (labels)
#    print (images)
    
    net = Net(batch_size)
    if load_checkpoint:
        net.load_state_dict(torch.load(SAVE_PATH))
        
    net.cuda()

    inputs= torch.from_numpy(img).float()/256
#        
#                # wrap them in Variable
#                
    inputs = Variable(inputs.cuda())
#    
    outputs = net.forward(inputs.view(1,1,112,112), False)
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

#    img = (inputs*256).data.cpu().int().numpy()[0,0]
    predict_boxes = outputs[:, (7*7*16+7*7*2):].contiguous().view(1, 7, 7, 2, 4)
    predict_confidence = outputs[:, 7*7*16:(7*7*16+7*7*2)].contiguous().view(1, 7, 7, 2)
    predict_class = outputs[:,:7*7*16].contiguous().view(1, 7, 7, 16)
    max_confidence = torch.max(predict_confidence, 3, keepdim = True)
    threshold = 0.2
    detect_ob = torch.ge(max_confidence[0], threshold).float()
    font = cv2.FONT_HERSHEY_PLAIN 
#####for test#######
#            for y in range(7):
#                for x in range(7):
#                    if labels[0,y,x,4]==1:
#                        xp, yp, w, h = labels[0,y,x,:4]
#                        lu = (int((x+xp)*16-w*112/2), int((y+yp)*16-h*112/2))
#                        rb = (int((x+xp)*16+w*112/2), int((y+yp)*16+h*112/2))
#                        cv2.rectangle(img, lu, rb, 200)
    for y in range(7):
        for x in range(7):
            if detect_ob.data.cpu().numpy()[0, y, x, 0] == 1:
                selection = max_confidence[1].data.cpu().numpy()[0,y,x,0]
                class_ = np.argmax(predict_class.data.cpu().numpy()[0,y,x])
                xp, yp, w, h = predict_boxes.data.cpu().numpy()[0,y,x,selection]
#                print((xp, yp, w, h))
#                print((x,y))
                lu = (int((x+xp)*16-w*112/2), int((y+yp)*16-h*112/2))
                rb = (int((x+xp)*16+w*112/2), int((y+yp)*16+h*112/2))


                color = 255
                cv2.putText(img,str(class_),(lu[0],lu[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
                cv2.rectangle(img, lu, rb, color)

#                print(lu)
#                print(rb)


    write_path = './test1result.jpg'
    cv2.imwrite(write_path,img)


    


    

