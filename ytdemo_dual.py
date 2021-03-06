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
from network_dual import Net
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
        self.num_classes = 39
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
        image_labels = np.reshape(image_labels, [self.num_cells, self.num_cells, 2*(self.num_classes+5)])


        if self.transform is not None:
            img = self.transform(img)

        return img, image_labels

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return self.file_count

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp_dual.pth'
    gpu_run = True
    if gpu_run:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 1
    cell_size = 28
    load_checkpoint= True
    num_cells = cell_size
    num_classes = 39
    img_size = 448
    box_size = 80

    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = 'validation_dual_label'
    img_path = 'validation_dual'
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

    if gpu_run:
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
            if gpu_run:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
#
            net.eval()
            if load_checkpoint:
                predicts = net.forward(inputs)
                results = predicts
                loss, accu = net.loss_function_vec(predicts, labels, 0.4, cal_accuracy=True)
                #accu_p = [accu[0][0].data.cpu().numpy(), accu[0][1].data.cpu().numpy(), accu[0][2].data.cpu().numpy(), accu[0][3].data.cpu().numpy()]
                print(accu)
            else:
                results = labels



            img = (inputs*256).data.cpu().int().numpy()[0,0]
            predict_class1 = results[:,:,:,5:num_classes+5]
            predict_confidence1 = results[:,:,:,4]
            predict_boxes1 = results[:,:,:,0:4]
            predict_class2 = results[:,:,:,num_classes+10:2*num_classes+10]
            predict_confidence2 = results[:,:,:,num_classes+9]
            predict_boxes2 = results[:,:,:,num_classes+5:num_classes+9]


            threshold = 0.4
            detect_ob1 = torch.ge(predict_confidence1[0], threshold).float()
            detect_ob2 = torch.ge(predict_confidence2[0], threshold).float()
            font = cv2.FONT_HERSHEY_PLAIN
            directory = os.path.dirname('../bounding_boxes/')
            if not os.path.exists(directory):
                os.makedirs(directory)

            for y in range(num_cells):
                for x in range(num_cells):
                    if detect_ob1.data.cpu().numpy()[y, x] == 1:
                        xp, yp, w, h = predict_boxes1.data.cpu().numpy()[0,y,x]
                        lu = (int((x+xp)*16-w*box_size/2), int((y+yp)*16-h*box_size/2))
                        rb = (int((x+xp)*16+w*box_size/2), int((y+yp)*16+h*box_size/2))
                        color = 255 #int(255 - img[lu[1], lu[0]])
                        class_1 = np.argmax(predict_class1.data.cpu().numpy()[0,y,x])
                        class_2 = np.argmax(predict_class2.data.cpu().numpy()[0,y,x])

                        #if class_ == gt_class:
                        #    color = 255
                        #else :
                        #    color = 100
                        if class_1 < 10:
                            cls_str = str(class_1)
                        elif class_1 < 36:
                            cls_str = chr(class_1 + 87)
                        elif class_1 == 36:
                            cls_str = "+"
                        elif class_1 == 37:
                            cls_str = "-"
                        elif class_1 == 38:
                            cls_str = "="
                        cv2.putText(img,cls_str,(lu[0],lu[1]), font, 1,(color,color,color), 1,cv2.LINE_AA, False)
                        cv2.rectangle(img, lu, rb, color)
                    if detect_ob2.data.cpu().numpy()[y, x] == 1:
                        xp, yp, w, h = predict_boxes2.data.cpu().numpy()[0,y,x]
                        lu = (int((x+xp)*16-w*box_size/2), int((y+yp)*16-h*box_size/2))
                        rb = (int((x+xp)*16+w*box_size/2), int((y+yp)*16+h*box_size/2))
                        color = 255 #int(255 - img[lu[1], lu[0]])

                        #if class_ == gt_class:
                        #    color = 255
                        #else :
                        #    color = 100
                        if class_2 < 10:
                            cls_str = str(class_2)
                        elif class_2 < 36:
                            cls_str = chr(class_2 + 87)
                        elif class_2 == 36:
                            cls_str = "+"
                        elif class_2 == 37:
                            cls_str = "-"
                        elif class_2 == 38:
                            cls_str = "="
                        cv2.putText(img,cls_str,(lu[0],lu[1]), font, 1,(color,color,color), 1,cv2.LINE_AA, False)
                        cv2.rectangle(img, lu, rb, color)



            write_path = '../bounding_boxes/'+str(i)+'.jpg'
            cv2.imwrite(write_path,img)
    print('Finished Marking!!')







