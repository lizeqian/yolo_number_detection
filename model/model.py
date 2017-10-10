import tensorflow as tf
import numpy as np
import copy

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
        
        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, 7, 7, 5+2])
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
        fc2_out = self.fc(fc1_out, fc2_w, fc2_b)
        fc2_out = tf.reshape(fc2_out, [-1, 7, 7, 12])
        
        return fc2_out
    
    def iou_calc(self, box0, box1): #x, y, w, h
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2.0,
                           boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2.0,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2.0,
                           boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, 0] - boxes2[:, :, :, 2] / 2.0,
                           boxes2[:, :, :, 1] - boxes2[:, :, :, 3] / 2.0,
                           boxes2[:, :, :, 0] + boxes2[:, :, :, 2] / 2.0,
                           boxes2[:, :, :, 1] + boxes2[:, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])
    

    def loss_function(self, predicts, labels):
        


        




