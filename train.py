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
from network import Net
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
        csv_path = csv_path
        self.img_paths = img_path
        image_labels = np.genfromtxt(csv_path, delimiter=',')
        image_labels.flatten()
        self.num_classes = 16
        image_labels = np.reshape(image_labels, [-1, 14, 14, self.num_classes+5])

        self.transform = transform
        self.labels=image_labels

    def __getitem__(self, index):
        image_addr = self.img_paths+'/'+str(index)+'.jpg'
        img = np.expand_dims(cv2.imread(image_addr,0), 0)
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return len(self.labels)

if __name__ == '__main__':
    SAVE_PATH = './checkpoint/cp1.bin'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    logger = Logger('./logs')
    batch_size = 50
    load_checkpoint= True

    print( '%s: calling main function ... ' % os.path.basename(__file__))
    csv_path = 'data16.csv'
    img_path = 'data16'
    validation_label = 'validation.csv'
    validation_data = 'validation'
    dataset = Rand_num(csv_path, img_path, 224, None)
    validationset = Rand_num(validation_label, validation_data, 224, None)
    sampler = RandomSampler(dataset)
    val_sampler = RandomSampler(validationset)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers=2)
    val_loader = DataLoader(validationset, batch_size = batch_size, sampler = val_sampler, shuffle = False, num_workers=2)

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

    optimizer = optim.Adam(net.parameters(), lr=0.00001)
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
                loss, accu = net.loss_function_vec(outputs, labels, 0.5, cal_accuracy=True)
                print (datetime.datetime.now())
                print ('Epoch %g'%(epoch))
                print(loss.data.cpu().numpy())
#                print(accu)
                #logger.scalar_summary('loss', loss.data.cpu().numpy(), epoch)
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
            optimizer.zero_grad()
            net.eval()
            outputs = net.forward(inputs)
            loss, _ = net.loss_function_vec(outputs, labels, 0.2)
            total_loss.append(loss.data.cpu().numpy())
        mean_loss = np.mean(total_loss)
        logger.scalar_summary('loss', mean_loss, epoch)
        print('val loss is %g'%(mean_loss))
        scheduler.step(mean_loss)


    torch.save(net.state_dict(), SAVE_PATH)
    print('Finished Training')

