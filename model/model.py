import tensorflow as tf
import numpy as np
import copy

class Net:
    def __init__(self):

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

    def conv2d(self, x, w, b):
        conv_ = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
        relu_ = tf.nn.relu(conv_+b)
        leaky_relu_ = tf.maximum(0.1*relu_, relu_)
        max_pool_ = tf.nn.max_pool(leaky_relu_, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
        return max_pool_

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def net_6_layers(self,image):
        layer1_out = self.conv2d(image, self.weight_variable(self.w_layer_1), self.bias_variable(self.b_layer_1))
        layer2_out = self.conv2d(layer1_out, self.weight_variable(self.w_layer_1), self.bias_variable(self.b_layer_1))
        layer3_out = self.conv2d(layer2_out, self.weight_variable(self.w_layer_2), self.bias_variable(self.b_layer_2))
        layer4_out = self.conv2d(layer3_out, self.weight_variable(self.w_layer_3), self.bias_variable(self.b_layer_3))
        layer5_out = self.conv2d(layer4_out, self.weight_variable(self.w_layer_4), self.bias_variable(self.b_layer_4))
        layer6_out = self.conv2d(layer5_out, self.weight_variable(self.w_layer_5), self.bias_variable(self.b_layer_5))


        




