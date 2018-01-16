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
    SAVE_PATH = './checkpoint/cp_28.pth'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 1
    cell_size = 28
    load_checkpoint= True
    num_cells = 28
    num_classes = 21
    img_size = 448

    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = 'validation28_label'
    img_path = 'validation28'
    dataset = Rand_num(csv_path, img_path, img_size, None)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=1)

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

    thld = np.arange(0,1,0.05)
    accu_tp=[]
    accu_fp=[]
    accu_iou=[]
    for epoch in range(1):
        for i, data in enumerate(loader, 0):
                # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float()/256, labels.float()
#
#                # wrap them in Variable
#
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#
            threshold=0.2
            net.eval()
            predicts = net.forward(inputs)
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
            gt_classes = labels[:, :, :, 5:]
            img = (inputs*256).data.cpu().int().numpy()[0,0]
            predict_class = predicts[:,:,:,:num_classes] #batch_size, cell_size, cell_size, num of class (class score)
            predict_confidence = predicts[:,:,:,num_classes:num_classes+2]#batch_size, cell_size, cell_size, num of boxes (box confidence)
            predict_boxes = predicts[:,:,:,num_classes+2:num_classes+10].contiguous().view(batch_size, cell_size, cell_size, 2, 4) # batch_size, cell_size, cell_size, boxes_num, 4 (box coordinate)
            #predict_boxes = outputs[:, (num_cells*num_cells*num_classes+num_cells*num_cells*2):].contiguous().view(1, num_cells, num_cells, 2, 4)
            #predict_confidence = outputs[:, num_cells*num_cells*num_classes:(num_cells*num_cells*num_classes+num_cells*num_cells*2)].contiguous().view(1, num_cells, num_cells, 2)
            #predict_class = outputs[:,:num_cells*num_cells*num_classes].contiguous().view(1, num_cells, num_cells, num_classes)
            max_confidence = torch.max(predict_confidence, 3, keepdim = True)
            threshold = 0.2
            detect_ob = torch.ge(max_confidence[0], threshold).float()
            font = cv2.FONT_HERSHEY_PLAIN
            directory = os.path.dirname('../bounding_boxes/')
            if not os.path.exists(directory):
                os.makedirs(directory)
#####for test#######
#            for y in range(7):
#                for x in range(7):
#                    if labels[0,y,x,4]==1:
#                        xp, yp, w, h = labels[0,y,x,:4]
#                        lu = (int((x+xp)*16-w*112/2), int((y+yp)*16-h*112/2))
#                        rb = (int((x+xp)*16+w*112/2), int((y+yp)*16+h*112/2))
#                        cv2.rectangle(img, lu, rb, 200)
            for y in range(num_cells):
                for x in range(num_cells):
                    if detect_ob.data.cpu().numpy()[0, y, x, 0] == 1:
                        selection = max_confidence[1].data.cpu().numpy()[0,y,x,0]
                        class_ = np.argmax(predict_class.data.cpu().numpy()[0,y,x])
                        gt_class = np.argmax(gt_classes.data.cpu().numpy()[0,y,x])
                        xp, yp, w, h = predict_boxes.data.cpu().numpy()[0,y,x,selection]
        #                print((xp, yp, w, h))
        #                print((x,y))
                        lu = (int((x+xp)*16-w*img_size/2), int((y+yp)*16-h*img_size/2))
                        rb = (int((x+xp)*16+w*img_size/2), int((y+yp)*16+h*img_size/2))

                        if class_ == gt_class:
                            color = 255
                        else :
                            color = 100
                            cv2.putText(img,str(class_),(lu[0],lu[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
                            cv2.putText(img,str(gt_class),(rb[0],rb[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
                            print(gt_class)
                        cv2.rectangle(img, lu, rb, color)

        #                print(lu)
        #                print(rb)


            write_path = '../bounding_boxes/'+str(i)+'.jpg'
            cv2.imwrite(write_path,img)
    print('Finished Marking')







