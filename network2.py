from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb
from IPython.core.debugger import Tracer
import math

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.num_classes = 16
        self.cell_size = 14
        self.img_size=224
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 26, 7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.cell_size* self.cell_size * 256, 4096*2)
        self.fc2 = nn.Linear(4096*2, self.cell_size*self.cell_size*(self.num_classes+10))
        self.offset = torch.arange(0,self.cell_size).expand(self.cell_size*2,self.cell_size).contiguous().view(2, self.cell_size, self.cell_size).permute(1, 2, 0).contiguous().view(1,self.cell_size,self.cell_size,2).expand(self.batch_size,self.cell_size,self.cell_size,2)
        self.offset = Variable(self.offset)
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_class = 2
        self.batchnorm1=nn.BatchNorm2d(32)
        self.batchnorm2=nn.BatchNorm2d(64)
        self.batchnorm3=nn.BatchNorm2d(128)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.batchnorm5=nn.BatchNorm2d(256)


    def forward(self, x):
        x = self.pool(F.leaky_relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.batchnorm3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.batchnorm4(self.conv4(x))))
        x = self.conv5(x)
        x = x.contiguous().view(-1, self.cell_size* self.cell_size * 26)
        #x = F.leaky_relu(self.fc1(x))
        #x = F.dropout(x)
        #x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def iou_calc(self, boxes1, boxes2): #x, y, w, h
        boxes1 = torch.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = boxes1.permute(1, 2, 3, 4, 0)

        boxes2 = torch.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = boxes2.permute(1, 2, 3, 4, 0)
#        print('boxes1')
#        print(boxes1)
#        print('boxes2')
#        print (boxes2)
        # calculate the left up point & right down point
        lu = torch.max(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rb = torch.min(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
#        print('lu')
#        print(lu)
#        print('rb')
#        print(rb)
        # intersection
        intersection = torch.max((rb - lu), Variable(torch.zeros(rb.size())))
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
#        print('intersection')
#        print(intersection)
#        print('inter_square')
#        print(inter_square)

        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
        square1 = torch.clamp(square1, 0.00001, self.img_size*self.img_size)
        square2 = torch.clamp(square2, 0.00001, self.img_size*self.img_size)
#        print('square1')
#        print(square1)
#        print('square2')
#        print(square2)
        union_square = square1 + square2 - inter_square
#        print('uniou_square')
#        print(union_square)
#        print('iou')
#        print(inter_square / union_square)
        result = inter_square / union_square
#        for i0 in range(100):
#            for i1 in range(7):
#                for i2 in range(7):
#                    for i3 in range(2):
#                        if math.isnan(result[i0,i1,i2,i3].data.cpu().numpy()):
#                            Tracer()()
        if math.isnan(torch.sum(result).data.cpu().numpy()):
            Tracer()()
        return result #shape = (batch_size, 7, 7, 2)

    def loss_function_vec(self, predicts, labels, threshold, cal_accuracy=False): #labels: [batch_size, cell_size_x, cell_size_y, 2+5] (x, y, w, h, C, p(c0), p(c1))

        predict_class = predicts[:,:self.cell_size*self.cell_size*self.num_classes].contiguous().view(self.batch_size, self.cell_size, self.cell_size, self.num_classes) #batch_size, cell_size, cell_size, num of class (class score)
        predict_confidence = predicts[:, self.cell_size*self.cell_size*self.num_classes:(self.cell_size*self.cell_size*self.num_classes+self.cell_size*self.cell_size*2)].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 2) #batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes = predicts[:, (self.cell_size*self.cell_size*self.num_classes+self.cell_size*self.cell_size*2):].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 2, 4) # batch_size, cell_size, cell_size, boxes_num, 4 (box coordinate)

        gt_object = labels[:, :, :, 4].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        gt_boxes = labels[:, :, :, 0:4].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1, 4)
        gt_boxes = gt_boxes.expand(self.batch_size, self.cell_size, self.cell_size, 2, 4)
        gt_classes = labels[:, :, :, 5:]
        predict_boxes_tran = torch.stack([(predict_boxes[:, :, :, :, 0] + self.offset) * 16,
                                          (predict_boxes[:, :, :, :, 1] + self.offset.permute(0, 2, 1, 3)) * 16,
                                           predict_boxes[:, :, :, :, 2] * self.img_size,
                                           predict_boxes[:, :, :, :, 3] * self.img_size])
        predict_boxes_tran = predict_boxes_tran.permute(1, 2, 3, 4, 0)

        gt_boxes_tran = torch.stack([(gt_boxes[:, :, :, :, 0] + self.offset) * 16,
                                     (gt_boxes[:, :, :, :, 1] + self.offset.permute(0, 2, 1, 3)) * 16,
                                      gt_boxes[:, :, :, :, 2] * self.img_size,
                                      gt_boxes[:, :, :, :, 3] * self.img_size])
        gt_boxes_tran = gt_boxes_tran.permute(1, 2, 3, 4, 0)

        gt_iou = self.iou_calc(predict_boxes_tran, gt_boxes_tran)
        max_iou = torch.max(gt_iou, 3, keepdim = True)
        max_iou=max_iou[0]
        object_mask = torch.mul(torch.ge(gt_iou,max_iou).float(), gt_object.float())
        noob_mask = Variable(torch.ones(object_mask.size())) - object_mask

        #class loss
        delta_p = gt_classes - predict_class
        delta_p_obj = delta_p * gt_object
        class_loss = self.lambda_class*torch.sum(delta_p_obj**2)/self.batch_size

        #coord loss
        coord_mask = object_mask.contiguous().view(self.batch_size, self.cell_size, self.cell_size, 2, 1)
        coord_delta = predict_boxes[:,:,:,:,:2] - gt_boxes[:,:,:,:,:2]
        coord_delta_mask = coord_delta * coord_mask
        size_delta = torch.sqrt(predict_boxes[:,:,:,:,2:]) - torch.sqrt(gt_boxes[:,:,:,:,2:])
        size_delta_mask = size_delta * coord_mask
        coord_loss = (torch.sum(coord_delta_mask**2) + \
                       torch.sum(size_delta_mask**2))/self.batch_size * self.lambda_coord

        #iou loss
        confidence_delta = predict_confidence - gt_iou

        iou_loss = (torch.sum((confidence_delta*object_mask)**2)+   \
                        self.lambda_noobj * torch.sum((confidence_delta*noob_mask)**2))/self.batch_size

#        print(class_loss)
#        print(coord_loss)
#        print(iou_loss)
        total_loss = class_loss + coord_loss + iou_loss
#        print (total_loss)
#        pdb.set_trace()

        accuracy = []
        if cal_accuracy is False:
            accuracy = [0,0,0]
        else:
            #############detection accuracy###############
            gt_noob = Variable(torch.ones(gt_object.size())) - gt_object
            threshold = threshold
            max_confidence = torch.max(predict_confidence, 3, keepdim = True)
            detect_iou = torch.ge(max_confidence[0], threshold).float()
            detect_iou_tp = detect_iou*gt_object
            detect_iou_fp = detect_iou*gt_noob
            detect_tp_accu = torch.sum(detect_iou_tp)/self.batch_size
            detect_fp_accu = torch.sum(detect_iou_fp)/self.batch_size
            detect_gt = torch.sum(gt_object)/self.batch_size
            detect_tp_accu = detect_tp_accu/detect_gt
            detect_fp_accu = detect_fp_accu/detect_gt
            accuracy.append(detect_tp_accu)
            accuracy.append(detect_fp_accu)
            #############iou accuracy#####################
            iou_accu = torch.sum(max_confidence[0]*detect_iou_tp)/self.batch_size
            iou_accu = iou_accu/detect_gt
            accuracy.append(iou_accu)
            #############class accuracy###################
            predicted_class = torch.max(predict_class, 3, keepdim = True)[1]
            groundtruth_classes = torch.max(gt_classes, 3, keepdim = True)[1]
            class_eq = (predicted_class==groundtruth_classes)
            class_hit = class_eq.float() * detect_iou *gt_object
            class_accu = torch.sum(class_hit.data.float())/torch.sum(gt_object)
            accuracy.append(class_accu)
        if math.isnan(total_loss.data.cpu().numpy()):
            for i0 in range(100):
                for i1 in range (self.cell_size):
                    for i2 in range (self.cell_size):
                        for i3 in range (2):
                            if math.isnan(confidence_delta[i0,i1,i2,i3].data.cpu().numpy()):
                                print ([i0,i1,i2,i3])
            Tracer()()
        return total_loss, accuracy



