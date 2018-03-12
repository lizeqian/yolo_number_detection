from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from IPython.core.debugger import Tracer
import math

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.num_classes = 0
        self.cell_size = 7
        self.img_size= 224
        self.batch_size = batch_size
        self.output_bits = self.num_classes+10
        self.conv1 = nn.Conv2d(1, 64, 7, padding=3)  #j=1, r=7
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1) #j=j*s=2, r=r+(k-1)*j=11
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1) #j=2, r=15
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1) #j=4, r=23
        self.conv5 = nn.Conv2d(256, 256, 7, padding=3) #j=16, r=71+16*6=167
        self.pool = nn.MaxPool2d(2, 2)
        self.offset = torch.arange(0,self.cell_size).expand(self.cell_size,self.cell_size).contiguous().view(1,self.cell_size, self.cell_size).expand(self.batch_size,self.cell_size,self.cell_size)
        self.offset = Variable(self.offset)
        self.lambda_coord = 3
        self.lambda_noobj = 0.1
        self.lambda_class = 1
        self.batchnorm1=nn.BatchNorm2d(64)
        self.batchnorm2=nn.BatchNorm2d(192)
        self.batchnorm3=nn.BatchNorm2d(384)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.batchnorm5=nn.BatchNorm2d(256)

        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256*7*7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.output_bits*7*7)
                )


    def forward(self, x):
        x = self.pool(F.leaky_relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.batchnorm3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.batchnorm4(self.conv4(x))))
        x = self.batchnorm5(self.conv5(x))
        x = x.contiguous().view(self.batch_size, -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        x = x.contiguous().view(self.batch_size, -1, self.cell_size, self.cell_size)
        x = x.permute(0,2,3,1)
        if math.isnan(torch.sum(x).data.cpu().numpy()):
            Tracer()()
        return x

    def iou_calc(self, boxes1, boxes2): #x, y, w, h
        boxes1 = torch.stack([boxes1[:, :, :, 0] - boxes1[:,  :, :, 2] / 2.0,
                           boxes1[:, :, :, 1] - boxes1[:, :,  :, 3] / 2.0,
                           boxes1[:, :, :, 0] + boxes1[:, :,  :, 2] / 2.0,
                           boxes1[:, :, :, 1] + boxes1[:, :,  :, 3] / 2.0])
        boxes1 = boxes1.permute(1, 2, 3, 0)

        boxes2 = torch.stack([boxes2[:, :, :, 0] - boxes2[:, :, :, 2] / 2.0,
                           boxes2[:, :, :, 1] - boxes2[:, :, :, 3] / 2.0,
                           boxes2[:, :, :, 0] + boxes2[:, :, :, 2] / 2.0,
                           boxes2[:, :, :, 1] + boxes2[:, :, :, 3] / 2.0])
        boxes2 = boxes2.permute(1, 2, 3, 0)
        # calculate the left up point & right down point
        lu = torch.max(boxes1[:, :, :, :2], boxes2[:, :, :, :2])
        rb = torch.min(boxes1[:, :, :, 2:], boxes2[:, :, :, 2:])
        # intersection
        intersection = torch.max((rb - lu), Variable(torch.zeros(rb.size())))
        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * \
                (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[:, :, :, 2] - boxes2[:, :, :, 0]) * \
                (boxes2[:, :, :, 3] - boxes2[:, :, :, 1])
        square1 = torch.clamp(square1, 0.00001, self.img_size*self.img_size)
        square2 = torch.clamp(square2, 0.00001, self.img_size*self.img_size)
        union_square = square1 + square2 - inter_square
        result = inter_square / union_square
        if math.isnan(torch.sum(result).data.cpu().numpy()):
            Tracer()()
        return result #shape = (batch_size, 7, 7, 2)

    def loss_function_vec(self, predicts, labels, threshold, cal_accuracy=False): #labels: [batch_size, cell_size_x, cell_size_y, 2+5] (x, y, w, h, C, p(c0), p(c1))

        #predict_class = predicts[:,:,:,:self.num_classes] #batch_size, cell_size, cell_size, num of class (class score)
        predict_confidence = predicts[:,:,:,4]#batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes = predicts[:,:,:,:4]
        gt_object = labels[:, :, :, 4]
        gt_boxes = labels[:, :, :, 0:4]
        #gt_classes = labels[:, :, :, 5:]
        predict_boxes_tran = torch.stack([(predict_boxes[:, :, :, 0] + self.offset) * 32,
                                          (predict_boxes[:, :, :, 1] + self.offset.permute(0, 2, 1)) * 32,
                                           predict_boxes[:, :, :, 2] * self.img_size,
                                           predict_boxes[:, :, :, 3] * self.img_size])
        predict_boxes_tran = predict_boxes_tran.permute(1, 2, 3, 0)

        gt_boxes_tran = torch.stack([(gt_boxes[:, :, :, 0] + self.offset) * 32,
                                     (gt_boxes[:, :, :, 1] + self.offset.permute(0, 2, 1)) * 32,
                                      gt_boxes[:, :, :, 2] * self.img_size,
                                      gt_boxes[:, :, :, 3] * self.img_size])
        gt_boxes_tran = gt_boxes_tran.permute(1, 2, 3, 0)

        gt_iou = self.iou_calc(predict_boxes_tran, gt_boxes_tran)
        #max_iou = torch.max(gt_iou, 3, keepdim = True)
        #max_iou=max_iou[0]
        #object_mask = torch.mul(torch.ge(gt_iou,max_iou).float(), gt_object.float())
        #noob_mask = Variable(torch.ones(object_mask.size())) - object_mask
        gt_noob = Variable(torch.ones(gt_object.size())) - gt_object
        #class loss
        #delta_p = gt_classes - predict_class
        #delta_p_obj = delta_p * gt_object
        #class_loss = self.lambda_class*torch.sum(delta_p_obj**2)/self.batch_size

        #coord loss
        coord_mask = gt_object.contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        coord_delta = predict_boxes[:,:,:,:2] - gt_boxes[:,:,:,:2]
        coord_delta_mask = coord_delta * coord_mask
        size_delta = torch.sqrt(predict_boxes[:,:,:,2:]) - torch.sqrt(gt_boxes[:,:,:,2:])
        size_delta_mask = size_delta * coord_mask
        coord_loss = torch.sum(coord_delta_mask**2)/self.batch_size * self.lambda_coord
        size_loss = torch.sum(size_delta_mask**2)/self.batch_size * self.lambda_coord
        #coo_loss = torch.sum(coord_delta_mask[0,6,6]**2)
        #size_loss = torch.sum(size_delta_mask[0,6,6]**2)

        #iou loss
        confidence_delta = predict_confidence - gt_iou
        #pred_con = predict_confidence[0,6,6]
        #gt_con = gt_iou[0,6,6]
        iou_loss = (torch.sum((confidence_delta*gt_object)**2)+   \
                        self.lambda_noobj * torch.sum((confidence_delta*gt_noob)**2))/self.batch_size
        #print("coord_loss")
        #print(coord_loss)
        #print("iou_loss")
        #print(iou_loss)
        total_loss = coord_loss + iou_loss + size_loss

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
            #predicted_class = torch.max(predict_class, 3, keepdim = True)[1]
            #groundtruth_classes = torch.max(gt_classes, 3, keepdim = True)[1]
            #class_eq = (predicted_class==groundtruth_classes)
            #class_hit = class_eq.float() * detect_iou *gt_object
            #class_accu = torch.sum(class_hit.data.float())/torch.sum(gt_object)
            #accuracy.append(class_accu)
        if math.isnan(total_loss.data.cpu().numpy()):
            for i0 in range(self.batch_size):
                for i1 in range (self.cell_size):
                    for i2 in range (self.cell_size):
                        for i3 in range (2):
                            if math.isnan(confidence_delta[i0,i1,i2,i3].data.cpu().numpy()):
                                print ([i0,i1,i2,i3])
            Tracer()()
        return total_loss, accuracy, coord_loss, size_loss, iou_loss



