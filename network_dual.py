from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from IPython.core.debugger import Tracer
import math

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.num_classes = 39
        self.cell_size = 28
        self.img_size=448
        self.batch_size = batch_size
        self.output_bits = self.num_classes*2+10
        self.conv1 = nn.Conv2d(1, 64, 7, padding=3)  #j=1, r=7, f=448
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2) #j=j*s=2, r=r+(k-1)*j=11, f=224
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1) #j=2, r=15, f=112
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1) #j=4, r=23, f=56
        self.conv5 = nn.Conv2d(256, self.output_bits, 5, padding=2) #j=16, r=71+16*6=167, f=28
        self.pool = nn.MaxPool2d(2, 2)
        self.offset = torch.arange(0,self.cell_size).expand(self.cell_size*2,self.cell_size).contiguous().view(1, self.cell_size, self.cell_size).expand(self.batch_size,self.cell_size,self.cell_size)
        self.offset = Variable(self.offset)
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_class = 2
        self.batchnorm1=nn.BatchNorm2d(64)
        self.batchnorm2=nn.BatchNorm2d(192)
        self.batchnorm3=nn.BatchNorm2d(384)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.batchnorm5=nn.BatchNorm2d(self.output_bits)


    def forward(self, x):
        x = self.pool(F.leaky_relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.batchnorm3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.batchnorm4(self.conv4(x))))
        x = self.batchnorm5(self.conv5(x))
        x = torch.sigmoid(x)
        x = x.permute(0,2,3,1)
        return x

    def iou_calc(self, boxes1, boxes2): #x, y, w, h
        boxes1 = torch.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2.0,
                           boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2.0,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2.0,
                           boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2.0])
        boxes1 = boxes1.permute(1, 2, 3, 0)

        boxes2 = torch.stack([boxes2[:, :, :, 0] - boxes2[:, :, :, 2] / 2.0,
                           boxes2[:, :, :, 1] - boxes2[:, :, :, 3] / 2.0,
                           boxes2[:, :, :, 0] + boxes2[:, :, :, 2] / 2.0,
                           boxes2[:, :, :, 1] + boxes2[:, :, :, 3] / 2.0])
        boxes2 = boxes2.permute(1, 2, 3, 0)
        lu = torch.max(boxes1[:, :, :, :2], boxes2[:, :, :, :2])
        rb = torch.min(boxes1[:, :, :, 2:], boxes2[:, :, :, 2:])
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
        return result #shape = (batch_size, 7, 7)

    def loss_function_vec(self, predicts, labels, threshold, cal_accuracy=False): #labels: [batch_size, cell_size_x, cell_size_y, 2+5] (x, y, w, h, C, p(c0), p(c1))

        predict_class1 = predicts[:,:,:,:self.num_classes] #batch_size, cell_size, cell_size, num of class (class score)
        predict_confidence1 = predicts[:,:,:,self.num_classes]#batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes1 = predicts[:,:,:,self.num_classes+1:self.num_classes+5]
        predict_class2 = predicts[:,:,:,self.num_classes+5:2*self.num_classes+5] #batch_size, cell_size, cell_size, num of class (class score)
        predict_confidence2 = predicts[:,:,:,2*self.num_classes+5]#batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes2 = predicts[:,:,:,2*self.num_classes+6:]


        gt_object1 = labels[:, :, :, 4].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        gt_boxes1 = labels[:, :, :, 0:4].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1, 4)
        #gt_boxes1 = gt_boxes1.expand(self.batch_size, self.cell_size, self.cell_size, 2, 4)
        gt_classes1 = labels[:, :, :, 5:self.num_classes+5]
        gt_object2 = labels[:, :, :, self.num_classes+9].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        gt_boxes2 = labels[:, :, :, self.num_classes+5:self.num_classes+9].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1, 4)
        #gt_boxes2 = gt_boxes2.expand(self.batch_size, self.cell_size, self.cell_size, 2, 4)
        gt_classes2 = labels[:, :, :, self.num_classes+10:]

        predict_boxes_tran1 = torch.stack([(predict_boxes1[:, :, :, 0] + self.offset) * 16,
                                          (predict_boxes1[:, :, :, 1] + self.offset.permute(0, 2, 1)) * 16,
                                           predict_boxes1[:, :, :, 2] * self.img_size,
                                           predict_boxes1[:, :, :, 3] * self.img_size])
        predict_boxes_tran1 = predict_boxes_tran1.permute(1, 2, 3, 0)
        predict_boxes_tran2 = torch.stack([(predict_boxes2[:, :, :, 0] + self.offset) * 16,
                                          (predict_boxes2[:, :, :, 1] + self.offset.permute(0, 2, 1)) * 16,
                                           predict_boxes2[:, :, :, 2] * self.img_size,
                                           predict_boxes2[:, :, :, 3] * self.img_size])
        predict_boxes_tran2 = predict_boxes_tran2.permute(1, 2, 3, 0)

        gt_boxes_tran1 = torch.stack([(gt_boxes1[:, :, :, 0] + self.offset) * 16,
                                     (gt_boxes1[:, :, :, 1] + self.offset.permute(0, 2, 1)) * 16,
                                      gt_boxes1[:, :, :, 2] * self.img_size,
                                      gt_boxes1[:, :, :, 3] * self.img_size])
        gt_boxes_tran1 = gt_boxes_tran1.permute(1, 2, 3, 0)
        gt_boxes_tran2 = torch.stack([(gt_boxes2[:, :, :, 0] + self.offset) * 16,
                                     (gt_boxes2[:, :, :, 1] + self.offset.permute(0, 2, 1)) * 16,
                                      gt_boxes2[:, :, :, 2] * self.img_size,
                                      gt_boxes2[:, :, :, 3] * self.img_size])
        gt_boxes_tran2 = gt_boxes_tran2.permute(1, 2, 3, 0)

        gt_iou1 = self.iou_calc(predict_boxes_tran1, gt_boxes_tran1)
        gt_iou2 = self.iou_calc(predict_boxes_tran2, gt_boxes_tran2)
        #max_iou = torch.max(gt_iou, 3, keepdim = True)
        #max_iou=max_iou[0]
        #object_mask = torch.mul(torch.ge(gt_iou,max_iou).float(), gt_object.float())
        #noob_mask = Variable(torch.ones(object_mask.size())) - object_mask
        gt_noob1 = Variable(torch.ones(gt_object1.size())) - gt_object1
        gt_noob2 = Variable(torch.ones(gt_object2.size())) - gt_object2

        #class loss
        delta_p1 = gt_classes1 - predict_class1
        delta_p_obj1 = delta_p1 * gt_object1
        delta_p2 = gt_classes2 - predict_class2
        delta_p_obj2 = delta_p2 * gt_object2
        class_loss = self.lambda_class*torch.sum(delta_p_obj1**2)/self.batch_size + self.lambda_class*torch.sum(delta_p_obj2**2)/self.batch_size

        #coord loss
        coord_mask1 = gt_object1.contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        coord_delta1 = predict_boxes1[:,:,:,:2] - gt_boxes1[:,:,:,:2]
        coord_delta_mask1 = coord_delta1 * coord_mask1
        size_delta1 = torch.sqrt(predict_boxes1[:,:,:,2:]) - torch.sqrt(gt_boxes1[:,:,:,2:])
        size_delta_mask1 = size_delta1 * coord_mask1
        coord_loss1 = (torch.sum(coord_delta_mask1**2) + \
                       torch.sum(size_delta_mask1**2))/self.batch_size * self.lambda_coord
        coord_mask2 = gt_object2.contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        coord_delta2 = predict_boxes2[:,:,:,:2] - gt_boxes2[:,:,:,:2]
        coord_delta_mask2 = coord_delta2 * coord_mask2
        size_delta2 = torch.sqrt(predict_boxes2[:,:,:,2:]) - torch.sqrt(gt_boxes2[:,:,:,2:])
        size_delta_mask2 = size_delta2 * coord_mask2
        coord_loss2 = (torch.sum(coord_delta_mask2**2) + \
                       torch.sum(size_delta_mask2**2))/self.batch_size * self.lambda_coord
        coord_loss = coord_loss1 + coord_loss2
        #iou loss
        confidence_delta1 = predict_confidence1 - gt_iou1
        confidence_delta2 = predict_confidence2 - gt_iou2

        iou_loss1 = (torch.sum((confidence_delta1*gt_object1)**2)+   \
                        self.lambda_noobj * torch.sum((confidence_delta1*gt_noob1)**2))/self.batch_size
        iou_loss2 = (torch.sum((confidence_delta2*gt_object2)**2)+   \
                        self.lambda_noobj * torch.sum((confidence_delta2*gt_noob2)**2))/self.batch_size

        iou_loss = iou_loss1 + iou_loss2

#        print(class_loss)
#        print(coord_loss)
#        print(iou_loss)
        total_loss = class_loss + coord_loss + iou_loss
#        print (total_loss)
#        pdb.set_trace()

        accuracy = []
#        if cal_accuracy is False:
#            accuracy = [0,0,0]
#        else:
#            #############detection accuracy###############
#            gt_noob = Variable(torch.ones(gt_object.size())) - gt_object
#            threshold = threshold
#            max_confidence = torch.max(predict_confidence, 3, keepdim = True)
#            detect_iou = torch.ge(max_confidence[0], threshold).float()
#            detect_iou_tp = detect_iou*gt_object  #tp
#            detect_iou_fp = detect_iou*gt_noob    #fp
#            sum_detect_tp = torch.sum(detect_iou_tp)
#            sum_detect_fp = torch.sum(detect_iou_fp)
#            sum_gtp = torch.sum(gt_object)
#            sum_detect = torch.sum(detect_iou)
#            accuracy.append([sum_detect_tp, sum_detect_fp, sum_gtp, sum_detect])
#            #detect_tp_accu = torch.sum(detect_iou_tp)/self.batch_size
#            #detect_fp_accu = torch.sum(detect_iou_fp)/self.batch_size
#            #detect_gt = torch.sum(gt_object)/self.batch_size
#            #detect_tp_accu = detect_tp_accu/detect_gt
#            #detect_fp_accu = detect_fp_accu/detect_gt
#            #accuracy.append(detect_tp_accu)
#            #accuracy.append(detect_fp_accu)
#            #############iou accuracy#####################
#            iou_accu = torch.sum(max_confidence[0]*detect_iou_tp)
#            iou_accu = iou_accu/sum_detect
#            accuracy.append(iou_accu)
#            #############class accuracy###################
#            predicted_class = torch.max(predict_class, 3, keepdim = True)[1]
#            groundtruth_classes = torch.max(gt_classes, 3, keepdim = True)[1]
#            class_eq = (predicted_class==groundtruth_classes)
#            class_hit = class_eq.float() * detect_iou *gt_object
#            class_accu = torch.sum(class_hit.data.float())/sum_detect
#            accuracy.append(class_accu)
#        if math.isnan(total_loss.data.cpu().numpy()):
#            for i0 in range(50):
#                for i1 in range (self.cell_size):
#                    for i2 in range (self.cell_size):
#                        for i3 in range (2):
#                            if math.isnan(confidence_delta[i0,i1,i2,i3].data.cpu().numpy()):
#                                print ([i0,i1,i2,i3])
#            Tracer()()
        return total_loss, accuracy



