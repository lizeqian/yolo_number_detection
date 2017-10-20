import tensorflow as tf
import numpy as np

class Net:
    def __init__(self, is_training=True):

        self.w_layer_1 = [7,7,1,32]
        self.w_layer_2 = [3,3,self.layer_1[3],64]
        self.w_layer_3 = [3,3,self.layer_2[3],128]
        self.w_layer_4 = [3,3,self.layer_3[3],256]
        self.w_layer_5 = [3,3,self.layer_4[3],512]
        self.w_layer_6 = [3,3,self.layer_5[3],1024]

        self.b_layer_1 = self.w_layer_1[3]
        self.b_layer_2 = self.w_layer_2[3]
        self.b_layer_3 = self.w_layer_3[3]
        self.b_layer_4 = self.w_layer_4[3]
        self.b_layer_5 = self.w_layer_5[3]
        self.b_layer_6 = self.w_layer_6[3]
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 1], name='images')
        self.logits = self.net_6_layers(self.images, is_training=is_training)

        self.batch_size = 100
        self.offset = np.transpose(np.reshape(np.array([np.arange(7)] * 7 * 2), (2, 7, 7)), (1, 2, 0))

        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, 7, 7, 2*5+2])
            self.loss_function(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            

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
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def loss_variable(self, shape):
        initial = tf.constant(0.0001, shape=shape)
        return tf.Variable(initial) 

    def net_6_layers(self,image,is_training=True):
        l1_w = self.weight_variable(self.w_layer_1)
        l1_b = self.bias_variable(self.b_layer_1)
        l2_w = self.weight_variable(self.w_layer_2)
        l2_b = self.bias_variable(self.b_layer_2)
        l3_w = self.weight_variable(self.w_layer_3)
        l3_b = self.bias_variable(self.b_layer_3)
        l4_w = self.weight_variable(self.w_layer_4)
        l4_b = self.bias_variable(self.b_layer_4)
        l5_w = self.weight_variable(self.w_layer_5)
        l5_b = self.bias_variable(self.b_layer_5)
        l6_w = self.weight_variable(self.w_layer_6)
        l6_b = self.bias_variable(self.b_layer_6)
        layer1_out = self.conv2d(image, l1_w, l1_b)
        layer2_out = self.conv2d(layer1_out, l2_w, l2_b)
        layer3_out = self.conv2d(layer2_out, l3_w, l3_b)
        layer4_out = self.conv2d(layer3_out, l4_w, l4_b)
        layer5_out = self.conv2d(layer4_out, l5_w, l5_b)
        layer6_out = self.conv2d(layer5_out, l6_w, l6_b)
        
        layer6_out_flat = tf.reshape(layer6_out, [-1, 7*7*self.w_layer_6[3]])
        fc1_w = self.weight_variable([7*7*self.w_layer_6[3], 4096])
        fc1_b = self.bias_variable([4096])
        fc2_w = self.weight_variable([4096, 7*7*12])
        fc2_b = self.bias_variable([7*7*12])
        fc1_out = self.fc(layer6_out_flat, fc1_w, fc1_b)
        fc2_out = self.fc(fc1_out, fc2_w, fc2_b)   #[7*7*num_classes, 7*7*num_boxes*5]
        #fc2_out = tf.reshape(fc2_out, [-1, 7, 7, 7])
        
        return fc2_out
    
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
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
        
        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
                
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def iou_calc_loop(self, boxes0, boxes1): #x, y, w, h
        inter_lu_x = tf.maximum(boxes0[0]-boxes0[2], boxes1[0]-boxes1[2])
        inter_lu_y = tf.maximum(boxes0[1]-boxes0[3], boxes1[1]-boxes1[3])
        inter_rb_x = tf.minimum(boxes0[0]+boxes0[2], boxes1[0]+boxes1[2])
        inter_rb_y = tf.minimum(boxes0[1]+boxes0[3], boxes1[1]+boxes1[3])
    
        inter_delta_x = tf.maximum(0.0, inter_rb_x - inter_lu_x)
        inter_delta_y = tf.maximum(0.0, inter_rb_y - inter_lu_y)
        
        inter_area = inter_delta_x*inter_delta_y
        iou = inter_area/(boxes0[2]*boxes0[3]+boxes1[2]+boxes1[3]-inter_area)
        
        return iou

    
    

    def loss_function(self, predicts, labels): #labels: [batch_size, cell_size_x, cell_size_y, 2+5] (x, y, w, h, C, p(c0), p(c1)) 
        predict_class = tf.reshape(predicts[:,:7*7*2], [self.batch_size, 7, 7, 2]) #batch_size, cell_size, cell_size, num of class (class score)
        predict_scales = tf.reshape(predicts[:, 7*7*2:(7*7*2+7*7*2)], [self.batch_size, 7, 7, 2]) #batch_size, cell_size, cell_size, num of boxes (box confidence)
        predict_boxes = tf.reshape(predicts[:, (7*7*2+7*7*2):], [self.batch_size, 7, 7, 2, 4]) # batch_size, cell_size, cell_size, boxes_num, 4 (box coordinate)
        
        #calculate loss
        class_loss = self.loss_variable([self.batch_size])
        iou_loss = self.loss_variable([self.batch_size])
        coord_loss = self.loss_variable([self.batch_size])
        
#       for i in range(self.batch_size):
#           for x in range(7):
#               for y in range(7):
#                   if labels[i, x, y, ] == 1:
#                       class_loss[i] += tf.square(labels[i, x, y, 5]-predict_class[i, x, y, 0]) + tf.square(labels[i, x, y, 6]-predict_class[i, x, y, 1])

        #calculate iou loss 

        label_boxes = labels[:,:,:,:4]


        for i in range(self.batch_size):
            for x in range(7):
                for y in range(7):
                    if labels[i, x, y, ] == 1:
                        class_loss[i] += tf.square(labels[i, x, y, 5]-predict_class[i, x, y, 0]) + tf.square(labels[i, x, y, 6]-predict_class[i, x, y, 1])
                    for b in range(2):
                        boxes0 = predict_boxes[i, x, y, b]
                        boxes1 = label_boxes[i, x, y]
                        true_iou = self.iou_calc_loop(boxes0, boxes1)
                        pred_confidence = predict_scales[i, x, y, b]
                        if labels[i, x, y, 4] == 1 and pred_confidence == tf.reduce_max(predict_scales[i,x, y, :]):
                            iou_loss[i] += tf.square(pred_confidence - true_iou)
                            coord_loss[i] += self.lambda_coord*(tf.square(predict_boxes[i, x, y, b, 0]-label_boxes[i, x, y, 0])+  \
                                                                tf.square(predict_boxes[i, x, y, b, 1]-label_boxes[i, x, y, 1])+  \
                                                                tf.square(tf.sqrt(predict_boxes[i, x, y, b, 2]-label_boxes[i, x, y, 2]))+  \
                                                                tf.square(tf.sqrt(predict_boxes[i, x, y, b, 3]-label_boxes[i, x, y, 3])))
                        else:
                            iou_loss[i] += self.lambda_noobj*tf.square(pred_confidence)

        #calculate coordinate loss 

#       for i in range(self.batch_size):
#           for x in range(7):
#               for y in range(7):
#                   for b in range(2):
#                       pred_confidence = predict_scales[i, x, y, b]
#                       if labels[i, x, y, 4] == 1 and pred_confidence == tf.reduce_max(predict_scales[i,x, y, :]):

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(iou_loss)
        tf.losses.add_loss(coord_loss)

