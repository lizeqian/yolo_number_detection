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
from network2 import Net
import torch.optim as optim
from logger import Logger
import torch.optim.lr_scheduler as lr_scheduler

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
        self.csv_paths = csv_path
        self.img_paths = img_path
        self.file_count = sum(len(files) for _, _, files in os.walk(img_path))
        self.num_classes = 16
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
        image_labels = np.reshape(image_labels, [28, 28, self.num_classes+5])


        if self.transform is not None:
            img = self.transform(img)

        return img, image_labels

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return self.file_count

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp_28.bin'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs_28')
    batch_size = 10
    load_checkpoint= False

    print (datetime.datetime.now())
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = 'data28_label'
    img_path = 'data28'
    validation_label = 'validation28_label'
    validation_data = 'validation28'
    dataset = Rand_num(csv_path, img_path, 448, None)
    validationset = Rand_num(validation_label, validation_data, 448, None)
    sampler = RandomSampler(dataset)
    val_sampler = RandomSampler(validationset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=2)
    val_loader = DataLoader(validationset, batch_size = batch_size, sampler = val_sampler, shuffle = False, num_workers=2)
    print (datetime.datetime.now())
    print ('dataset comp')

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

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    for epoch in range(2000):
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float()/256, labels.float()

            # wrap them in Variable

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #print (inputs)
            net.train()
            outputs = net.forward(inputs)
            loss, _ = net.loss_function_vec(outputs, labels, 0.5)
            loss.backward()
            optimizer.step()
            # print statistics
            #running_loss += loss.data[0]
            if epoch % 1 == 0 and i == 0:
                net.eval()
                outputs = net.forward(inputs)
                loss, accu = net.loss_function_vec(outputs, labels, 0.2, cal_accuracy=True)
                print (datetime.datetime.now())
                print ('Epoch %g'%(epoch))
                print(loss.data.cpu().numpy())
#                print(accu)
                logger.scalar_summary('training loss', loss.data.cpu().numpy(), epoch)
                #logger.scalar_summary('Accuracy detection TP', accu[0].data.cpu().numpy(), epoch)
                #logger.scalar_summary('Accuracy detection FP', accu[1].data.cpu().numpy(), epoch)
                #logger.scalar_summary('Accuracy IOU', accu[2].data.cpu().numpy(), epoch)
#                logger.scalar_summary('Accuracy Class', accu[3].data.cpu().numpy(), epoch)
            if epoch % 1 == 0 and i==0:
                torch.save(net.state_dict(), SAVE_PATH)
        total_loss=[]
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.float()/256, labels.float()
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile = True)
            net.eval()
            outputs = net.forward(inputs)
            loss, _ = net.loss_function_vec(outputs, labels, 0.2)
            total_loss.append(loss.data.cpu().numpy())
        mean_loss = np.mean(total_loss)
        print('val loss is %g'%(mean_loss))
        logger.scalar_summary('validation loss', mean_loss, epoch)
        scheduler.step(mean_loss)


    torch.save(net.state_dict(), SAVE_PATH)
    print('Finished Training')

