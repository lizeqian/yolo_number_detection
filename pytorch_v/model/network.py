from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32, 7)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7* 7 * 256, 4096)
        self.fc2 = nn.Linear(4096, 7*7*12)
        self.offset = torch.arange(0,7).expand(14,7).contiguous().view(2, 7, 7).permute(1, 2, 0).contiguous().view(1,7,7,2).expand(self.batch_size,7,7,2)

    def forward(self, x, is_training):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = x.contiguous().view(-1, 7* 7 * 256)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=is_training)
        x = self.fc2(x)
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
        
        # calculate the left up point & right down point
        lu = torch.max(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = torch.max(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
        
        # intersection
        intersection = torch.max(rd - lu, 0.0)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
                
        union_square = torch.max(square1 + square2 - inter_square, 1e-10)
        return inter_square / union_square #shape = (batch_size, 7, 7, 2)
        
    
    def loss_function_vec(self, predicts, labels): #labels: [batch_size, cell_size_x, cell_size_y, 2+5] (x, y, w, h, C, p(c0), p(c1)) 
        predict_class = predicts[:,:7*7*2].contiguous().view(self.batch_size, 7, 7, 2) #batch_size, cell_size, cell_size, num of class (class score)
        predict_confidence = predicts[:, 7*7*2:(7*7*2+7*7*2)].contiguous().view(self.batch_size, 7, 7, 2) #batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes = predicts[:, (7*7*2+7*7*2):].contiguous().view(self.batch_size, 7, 7, 2, 4) # batch_size, cell_size, cell_size, boxes_num, 4 (box coordinate)
        
        gt_object = labels[:, :, :, 4].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1)
        gt_boxes = labels[:, :, :, 0:4].contiguous().view(self.batch_size, self.cell_size, self.cell_size, 1, 4)
        gt_boxes = gt_boxes.expand(1, 1, 1, 2, 1)
        gt_classes = labels[:, :, :, 5:]
        
        predict_boxes_tran = torch.stack([(predict_boxes[:, :, :, :, 0] + self.offset) * 16,
                                          (predict_boxes[:, :, :, :, 1] + self.offset.permute(0, 2, 1, 3)) * 16,
                                           predict_boxes[:, :, :, :, 2] * 112,
                                           predict_boxes[:, :, :, :, 3] * 112])
        predict_boxes_tran = predict_boxes_tran.permute(1, 2, 3, 4, 0)
        
        gt_boxes_tran = torch.stack([(gt_boxes[:, :, :, :, 0] + self.offset) * 16,
                                     (gt_boxes[:, :, :, :, 1] + self.offset.permute(0, 2, 1, 3)) * 16,
                                      gt_boxes[:, :, :, :, 2] * 112,
                                      gt_boxes[:, :, :, :, 3] * 112])
        gt_boxes_tran = gt_boxes_tran.permute(1, 2, 3, 4, 0)
        
        gt_iou = self.iou_calc(predict_boxes_tran, gt_boxes_tran)
        max_iou = torch.max(gt_iou, 3, keep_dims = True)
        object_mask = (gt_iou >= max_iou).float() * gt_object
        noob_mask = torch.from_numpy(np.ones_like(object_mask)) - object_mask
        
        #class loss
        delta_p = gt_classes - predict_class
        delta_p_obj = delta_p * gt_object
        class_loss = torch.sum(delta_p_obj**2)/self.batch_size
        
        #coord loss
        coord_mask = object_mask.contiguous().view(self.batch_size, 7, 7, 2, 1) 
        coord_delta = predict_boxes[:,:,:,:,:2] - gt_boxes[:,:,:,:,:2]
        coord_delta_mask = coord_delta * coord_mask
        size_delta = torch.sqrt(predict_boxes[:,:,:,:,2:]) - torch.sqrt(gt_boxes[:,:,:,:,2:])
        size_delta_mask = size_delta * coord_mask
        coord_loss = (torch.sum(coord_delta_mask**2) + \
                       torch.sum(size_delta_mask**2))/self.batch_size * self.lambda_coord
                                                
        #iou loss
        confidence_delta = predict_confidence - gt_iou
        
        iou_loss = (torch.sum(confidence_delta*object_mask**2)+   \
                        self.lambda_noobj * torch.sum(confidence_delta*noob_mask**2))/self.batch_size
        
                    
        total_loss = class_loss + coord_loss + iou_loss
        #Accuracy IOU
#        pr_iou = torch.max(gt_iou, 3, keep_dims = True)
#        pr_class = torch.max(predict_class, 3, keep_dims = True)
#        pr_c = pr_iou * pr_class
#        pr_c_nms = self.non_max_suppression(pr_c, 3) #Predicted boxes
        
#        sel_iou = pr_iou*pr_c_nms
#
#        sel_iou_mask = sel_iou*gt_object
#        accu_iou = tf.reduce_mean(tf.reduce_sum(sel_iou_mask, axis=[1,2,3]))
#        
#        class accuracy
#        class_predicted = tf.where(tf.greater(predict_class[:,:,:,0], predict_class[:,:,:,1]), tf.ones_like(predict_class[:,:,:,0]), tf.zeros_like(predict_class[:,:,:,0]))
#        class_labeled = tf.where(tf.greater(gt_classes[:,:,:,0], gt_classes[:,:,:,1]), tf.ones_like(gt_classes[:,:,:,0]), tf.zeros_like(gt_classes[:,:,:,0]))
#        accu_class = tf.reduce_mean(tf.reduce_sum(tf.where(tf.equal(class_predicted, class_labeled), tf.ones_like(class_predicted), tf.zeros_like(class_predicted))*tf.reshape(gt_object, [self.batch_size, self.cell_size, self.cell_size])*tf.reshape(pr_c_nms, [self.batch_size, self.cell_size, self.cell_size]), axis=[1,2]))
#        
#        #Detection accuracy
#        accu_detect = tf.reduce_mean(tf.reduce_sum(pr_c_nms*gt_object, axis=[1,2,3]))
#        
#        #False positive accuracy
#        gt_noob = tf.where(tf.equal(gt_object, 0), tf.ones_like(gt_object), tf.zeros_like(gt_object))
#        accu_fp = tf.reduce_mean(tf.reduce_sum(pr_c_nms*gt_noob, axis=[1,2,3]))
#        
#        tf.losses.add_loss(class_loss)
#        tf.losses.add_loss(coord_loss)
#        tf.losses.add_loss(iou_loss)
#        
#        self.variable_summaries(class_loss, 'class_loss')
#        self.variable_summaries(coord_loss, 'coord_loss')
#        self.variable_summaries(iou_loss, 'iou_loss')
#        
        return total_loss #tf.losses.get_total_loss(), accu_iou, accu_class, accu_detect, accu_fp