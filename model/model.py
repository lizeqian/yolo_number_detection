import tensorflow as tf
import numpy as np

class Net:
    def __init__(self, is_training, batch_size):

        self.w_layer_1 = [7,7,1,32]
        self.w_layer_2 = [3,3,self.w_layer_1[3],64]
        self.w_layer_3 = [3,3,self.w_layer_2[3],128]
        self.w_layer_4 = [3,3,self.w_layer_3[3],256]
        self.w_layer_5 = [3,3,self.w_layer_4[3],512]
        self.w_layer_6 = [3,3,self.w_layer_5[3],1024]

        self.b_layer_1 = [self.w_layer_1[3]]
        self.b_layer_2 = [self.w_layer_2[3]]
        self.b_layer_3 = [self.w_layer_3[3]]
        self.b_layer_4 = [self.w_layer_4[3]]
        self.b_layer_5 = [self.w_layer_5[3]]
        self.b_layer_6 = [self.w_layer_6[3]]

        self.batch_size = batch_size

        self.lambda_noobj = 0.5
        self.lambda_coord = 2
        
        self.image_size = 112
        self.cell_size = 7
        
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(7)] * 7 * 2),
            (2, 7, 7)), (1, 2, 0))
        
    def variable_summaries(self, var, scope): 
        with tf.name_scope(scope):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))       

    def conv2d(self, x, w, b):
        conv_ = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
        relu_ = tf.nn.relu(conv_+b)
        leaky_relu_ = tf.maximum(0.1*relu_, relu_)
        max_pool_ = tf.nn.max_pool(leaky_relu_, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
        return max_pool_
    
    def fc(self, x, w, b):
        relu_ = tf.nn.relu(tf.matmul(x, w) + b)
        return tf.maximum(0.1*relu_, relu_)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.00001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def net_4_layers(self,image,keep_prob):
        l1_w = self.weight_variable(self.w_layer_1)
        l1_b = self.bias_variable(self.b_layer_1)
        l2_w = self.weight_variable(self.w_layer_2)
        l2_b = self.bias_variable(self.b_layer_2)
        l3_w = self.weight_variable(self.w_layer_3)
        l3_b = self.bias_variable(self.b_layer_3)
        l4_w = self.weight_variable(self.w_layer_4)
        l4_b = self.bias_variable(self.b_layer_4)
#        l5_w = self.weight_variable(self.w_layer_5)
#        l5_b = self.bias_variable(self.b_layer_5)
#        l6_w = self.weight_variable(self.w_layer_6)
#        l6_b = self.bias_variable(self.b_layer_6)
        layer1_out = self.conv2d(image, l1_w, l1_b)
        layer2_out = self.conv2d(layer1_out, l2_w, l2_b)
        layer3_out = self.conv2d(layer2_out, l3_w, l3_b)
        layer4_out = self.conv2d(layer3_out, l4_w, l4_b)
#        layer5_out = self.conv2d(layer4_out, l5_w, l5_b)
#        layer6_out = self.conv2d(layer5_out, l6_w, l6_b)
        self.variable_summaries(l1_w, 'l2_w')
        self.variable_summaries(l1_b, 'l2_b')
        self.variable_summaries(l2_w, 'l2_w')
        self.variable_summaries(l2_b, 'l2_b')
        self.variable_summaries(l2_w, 'l2_w')
        self.variable_summaries(l2_b, 'l2_b')
        self.variable_summaries(l2_w, 'l2_w')
        self.variable_summaries(l2_b, 'l2_b')
        self.variable_summaries(layer1_out, 'layer1_out')
        self.variable_summaries(layer2_out, 'layer2_out')
        self.variable_summaries(layer3_out, 'layer3_out')
        self.variable_summaries(layer4_out, 'layer4_out')
#        layer6_out_flat = tf.reshape(layer6_out, [-1, 7*7*self.w_layer_6[3]])
        layer4_out_flat = tf.reshape(layer4_out, [-1, 7*7*self.w_layer_4[3]])
        
        fc1_w = self.weight_variable([7*7*self.w_layer_4[3], 4096])
        fc1_b = self.bias_variable([4096])
        fc1_out = self.fc(layer4_out_flat, fc1_w, fc1_b)       
        self.variable_summaries(fc1_out, 'fc1_out')
        fc1_drop = tf.nn.dropout(fc1_out, keep_prob)
        fc2_w = self.weight_variable([4096, 7*7*12])
        fc2_b = self.bias_variable([7*7*12])
        fc2_out = tf.matmul(fc1_drop, fc2_w) + fc2_b
        self.variable_summaries(fc2_out, 'fc2_out')
        #fc2_out = self.fc(fc1_out, fc2_w, fc2_b)   #[7*7*num_classes, 7*7*num_boxes*5]
        #fc2_out = tf.reshape(fc2_out, [-1, 7, 7, 7])
        #print (fc2_out) 
        return fc2_out #[?, 7*7*12]
    
    def non_max_suppression(self, window, window_size):
        # input: B x W x H x C
        pooled = tf.nn.max_pool(window, ksize=[1, window_size, window_size, 1], strides=[1,1,1,1], padding='SAME')
        output = tf.where(tf.equal(window, pooled), tf.ones_like(window), tf.zeros_like(window))
    
        # NOTE: if input has negative values, the suppressed values can be higher than original
        return output # output: B X W X H x C

    
    def iou_calc(self, boxes1, boxes2): #x, y, w, h
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])
        
        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2],- boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
        
        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
                
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0) #shape = (batch_size, 7, 7, 2)
        
    
    def loss_function_vec(self, predicts, labels): #labels: [batch_size, cell_size_x, cell_size_y, 2+5] (x, y, w, h, C, p(c0), p(c1)) 
        predict_class = tf.reshape(predicts[:,:7*7*2], [self.batch_size, 7, 7, 2]) #batch_size, cell_size, cell_size, num of class (class score)
        predict_confidence = tf.reshape(predicts[:, 7*7*2:(7*7*2+7*7*2)], [self.batch_size, 7, 7, 2]) #batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes = tf.reshape(predicts[:, (7*7*2+7*7*2):], [self.batch_size, 7, 7, 2, 4]) # batch_size, cell_size, cell_size, boxes_num, 4 (box coordinate)
        
        gt_object = tf.reshape(labels[:, :, :, 4], [self.batch_size, self.cell_size, self.cell_size, 1])
        gt_boxes = tf.reshape(labels[:, :, :, 0:4], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
        gt_boxes = tf.tile(gt_boxes, [1, 1, 1, 2, 1])
        gt_classes = labels[:, :, :, 5:]
        
        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, 7, 7, 2])
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) * 16,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) * 16,
                                           tf.square(predict_boxes[:, :, :, :, 2]) * 112,
                                           tf.square(predict_boxes[:, :, :, :, 3]) * 112])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])
        
        gt_boxes_tran = tf.stack([(gt_boxes[:, :, :, :, 0] + offset) * 16,
                                           (gt_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) * 16,
                                           tf.square(gt_boxes[:, :, :, :, 2]) * 112,
                                           tf.square(gt_boxes[:, :, :, :, 3]) * 112])
        gt_boxes_tran = tf.transpose(gt_boxes_tran, [1, 2, 3, 4, 0])
        
        gt_iou = self.iou_calc(predict_boxes_tran, gt_boxes_tran)
        max_iou = tf.reduce_max(gt_iou, axis = 3, keep_dims = True)
        object_mask = tf.cast((gt_iou >= max_iou), tf.float32) * gt_object
        noob_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        
        #class loss
        delta_p = gt_classes - predict_class
        delta_p_obj = delta_p * gt_object
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(delta_p_obj), axis=[1,2,3]))
        
        #coord loss
        coord_mask = tf.expand_dims(object_mask, 4)
        coord_delta = predict_boxes[:,:,:,:,:2] - gt_boxes[:,:,:,:,:2]
        coord_delta_mask = coord_delta * coord_mask
        size_delta = tf.sqrt(predict_boxes[:,:,:,:,2:]) - tf.sqrt(gt_boxes[:,:,:,:,2:])
        size_delta_mask = size_delta * coord_mask
        coord_loss = (tf.reduce_mean(tf.reduce_sum(tf.square(coord_delta_mask), axis=[1,2,3,4])) + \
                      tf.reduce_mean(tf.reduce_sum(tf.square(size_delta_mask), axis=[1,2,3,4])))*self.lambda_coord
                                                
        #iou loss
        confidence_delta = predict_confidence - gt_iou
        
        iou_loss = (tf.reduce_mean(tf.reduce_sum(tf.square(confidence_delta*object_mask), axis=[1,2,3])))+   \
                        self.lambda_noobj * (tf.reduce_mean(tf.reduce_sum(tf.square(confidence_delta*noob_mask), axis=[1,2,3])))
        
        #Accuracy IOU
        pr_iou = tf.reduce_max(gt_iou, axis = 3, keep_dims = True)
        pr_class = tf.reduce_max(predict_class, axis = 3, keep_dims = True)
        pr_c = pr_iou * pr_class
        pr_c_nms = self.non_max_suppression(pr_c, 3) #Predicted boxes
        
        sel_iou = pr_iou*pr_c_nms

        sel_iou_mask = sel_iou*gt_object
        accu_iou = tf.reduce_mean(tf.reduce_sum(sel_iou_mask, axis=[1,2,3]))
        
        #class accuracy
        class_predicted = tf.where(tf.greater(predict_class[:,:,:,0], predict_class[:,:,:,1]), tf.ones_like(predict_class[:,:,:,0]), tf.zeros_like(predict_class[:,:,:,0]))
        class_labeled = tf.where(tf.greater(gt_classes[:,:,:,0], gt_classes[:,:,:,1]), tf.ones_like(gt_classes[:,:,:,0]), tf.zeros_like(gt_classes[:,:,:,0]))
        accu_class = tf.reduce_mean(tf.reduce_sum(tf.where(tf.equal(class_predicted, class_labeled), tf.ones_like(class_predicted), tf.zeros_like(class_predicted))*tf.reshape(gt_object, [self.batch_size, self.cell_size, self.cell_size])*tf.reshape(pr_c_nms, [self.batch_size, self.cell_size, self.cell_size]), axis=[1,2]))
        
        #Detection accuracy
        accu_detect = tf.reduce_mean(tf.reduce_sum(pr_c_nms*gt_object, axis=[1,2,3]))
        
        #False positive accuracy
        gt_noob = tf.where(tf.equal(gt_object, 0), tf.ones_like(gt_object), tf.zeros_like(gt_object))
        accu_fp = tf.reduce_mean(tf.reduce_sum(pr_c_nms*gt_noob, axis=[1,2,3]))
        
        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(coord_loss)
        tf.losses.add_loss(iou_loss)
        
        self.variable_summaries(class_loss, 'class_loss')
        self.variable_summaries(coord_loss, 'coord_loss')
        self.variable_summaries(iou_loss, 'iou_loss')
        
        return tf.losses.get_total_loss(), accu_iou, accu_class, accu_detect, accu_fp