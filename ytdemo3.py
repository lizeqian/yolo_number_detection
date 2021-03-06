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
import vgg
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from network3 import Net
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
        self.num_classes = 0
        self.num_cells = 7

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
    SAVE_PATH = './checkpoint/cp_3.pth'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 1
    cell_size = 7
    load_checkpoint= True
    num_cells = cell_size
    num_classes = 0
    img_size = 224

    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = 'validation_eq_label'
    img_path = 'validation_eq'
    dataset = Rand_num(csv_path, img_path, img_size, None)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=1)

#    dataiter = iter(loader)
#    images, labels = dataiter.next()
#    print (images)
#    images=tensor_to_img(images)
#    print (labels)
#    print (images)

    #net = Net(batch_size)
    vgg_model = vgg.vgg19_bn(num_classes=7*7*10)
    if load_checkpoint:
        vgg_model.load_state_dict(torch.load(SAVE_PATH))

    vgg_model.cuda()

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
            vgg_model.eval()
            outputs = vgg_model.forward(inputs)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.contiguous().view(batch_size, -1, 7, 7)
            predicts = outputs.permute(0,2,3,1)
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
            #gt_classes = labels[:, :, :, 5:]
            img = (inputs*256).data.cpu().int().numpy()[0,0]
            #predict_class = predicts[:,:,:,:num_classes] #batch_size, cell_size, cell_size, num of class (class score)
            predict_confidence = predicts[:,:,:,:2]#batch_size, cell_size, cell_size, num of boxes (box confidence)
            predict_boxes = predicts[:,:,:,2:].contiguous().view(batch_size, cell_size, cell_size,2,4) # batch_size, cell_size, cell_size, boxes_num, 4 (box coordinate)
            #predict_boxes = outputs[:, (num_cells*num_cells*num_classes+num_cells*num_cells*2):].contiguous().view(1, num_cells, num_cells, 2, 4)
            #predict_confidence = outputs[:, num_cells*num_cells*num_classes:(num_cells*num_cells*num_classes+num_cells*num_cells*2)].contiguous().view(1, num_cells, num_cells, 2)
            #predict_class = outputs[:,:num_cells*num_cells*num_classes].contiguous().view(1, num_cells, num_cells, num_classes)
            max_confidence = torch.max(predict_confidence, 3, keepdim = True)
            print(max_confidence[0])
            threshold = 0.2
            detect_ob = torch.ge(max_confidence[0], threshold).float()
            font = cv2.FONT_HERSHEY_PLAIN
            directory = os.path.dirname('bounding_boxes/')
            #print(predict_confidence[0,:,:, 0])
            if load_checkpoint is False:
                detect_ob = labels[:,:,:,4].contiguous().view(labels.size(0), labels.size(1), labels.size(2), 1)
                predict_boxes = labels[:,:,:,:4]

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
                    if detect_ob.data.cpu().numpy()[0, y, x, 0] == 1 and load_checkpoint:
                        selection = max_confidence[1].data.cpu().numpy()[0,y,x,0]
                        xp, yp, w, h = predict_boxes.data.cpu().numpy()[0,y,x,selection]
                        lu = (int((x+xp)*32-w*img_size/2), int((y+yp)*32-h*img_size/2))
                        rb = (int((x+xp)*32+w*img_size/2), int((y+yp)*32+h*img_size/2))
                        color = 255 #int(255 - img[lu[1], lu[0]])
                        #if class_ == gt_class:
                        #    color = 255
                        #else :
                        #    color = 100
                        #if class_ < 10:
                        #    cls_str = str(class_)
                        #elif class_ == 11:
                        #    cls_str = "x"
                        #elif class_ == 12:
                        #    cls_str = "y"
                        #elif class_ == 13:
                        #    cls_str = "z"
                        #elif class_ == 14:
                        #    cls_str = "a"
                        #elif class_ == 15:
                        #    cls_str = "b"
                        #elif class_ == 16:
                        #    cls_str = "m"
                        #elif class_ == 17:
                        #    cls_str = "n"
                        #elif class_ == 18:
                        #    cls_str = "+"
                        #elif class_ == 19:
                        #    cls_str = "-"
                        #elif class_ == 20:
                        #    cls_str = "="
                        #cv2.putText(img,cls_str,(lu[0],lu[1]), font, 1,(color,color,color), 1,cv2.LINE_AA, False)
                        #    cv2.putText(img,str(gt_class),(rb[0],rb[1]), font, 1,(255,255,255), 1,cv2.LINE_AA, False)
                        #    print(gt_class)
                        cv2.rectangle(img, lu, rb, color)
                    elif detect_ob.data.cpu().numpy()[0, y, x, 0] == 1:
                        xp, yp, w, h = predict_boxes.data.cpu().numpy()[0,y,x]
                        lu = (int((x+xp)*32-w*img_size/2), int((y+yp)*32-h*img_size/2))
                        rb = (int((x+xp)*32+w*img_size/2), int((y+yp)*32+h*img_size/2))
                        color = 255 #int(255 - img[lu[1], lu[0]])
                        cv2.rectangle(img, lu, rb, color)

        #                print(lu)
        #                print(rb)


            write_path = 'bounding_boxes/'+str(i)+'.jpg'
            cv2.imwrite(write_path,img)
    print('Finished Marking')







