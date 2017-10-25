import tensorflow as tf
import numpy as np
import datetime
import os
import argparse
import glob

from model.model import Net

def read_image(filename_queue):
    label = filename_queue[1]
    file_contents = tf.read_file(filename_queue[0])
    image = tf.image.decode_jpeg(file_contents)
    return image, label

class Solver:
    def __init__(self, num_epochs, batch_size,  net):
        self.num_epochs = num_epochs
        self.net = net
        self.batch_size = batch_size
    
    def load_data(self, image_dir, label_dir):
        image_path = image_dir+'/*.png'
        image_addrs = glob.glob(image_path)  
        image_labels = np.genfromtxt(label_dir, delimiter=',')
        image_labels = np.reshape(image_labels, [-1, 7, 7, 7])
        filename_queue = tf.train.slice_input_producer([image_addrs, image_labels], num_epochs=self.num_epochs, shuffle=False)
        image, label = read_image(filename_queue)
        image.set_shape((112, 112, 1))
        image_batch, label_batch = tf.train.batch([image, label], batch_size=self.batch_size)
        return image_batch, label_batch
        
    def train(self):
        sess = tf.InteractiveSession()
        
        image_batch, label_batch = self.load_data('detection_training', 'detection_training.csv')
        print ("Data loaded")
        image = tf.placeholder(tf.float32, shape=[None, 112,112,1])
        label = tf.placeholder(tf.float32, shape=[None, 7, 7, 7])
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.placeholder(tf.int32)
        net_out = self.net.net_4_layers(image, keep_prob)
        total_loss, accu_iou, accu_class, accu_detect, acc_fp = self.net.loss_function_vec(net_out, label)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('accu_iou', accu_iou)
        tf.summary.scalar('accu_class', accu_class)
        tf.summary.scalar('accu_detect', accu_detect)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./log/', sess.graph)
        print ("Network built")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        starter_learning_rate = 5e-7
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 300, 0.7, staircase=False)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)   
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        saver = tf.train.Saver()
        

        sess.run(init_op)
        if False:
            saver.restore(sess, "./tmp/My_Model.ckpt")
            print("Model restored.")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print ("Begin Training")
        for epoch in range(0, self.num_epochs):
            image_tensor,image_label = sess.run([image_batch,label_batch])
            summary, _ = sess.run([merged, train_step], feed_dict={image: image_tensor, label: image_label, keep_prob: 0.5, global_step: epoch})
            if epoch%100 == 0:
                loss = total_loss.eval(feed_dict={image: image_tensor, label: image_label, keep_prob: 1.0, global_step: epoch})
                accuracy_iou = accu_iou.eval(feed_dict={image: image_tensor, label: image_label, keep_prob: 1.0, global_step: epoch})
                accuracy_class = accu_class.eval(feed_dict={image: image_tensor, label: image_label, keep_prob: 1.0, global_step: epoch})
                accuracy_detect= accu_detect.eval(feed_dict={image: image_tensor, label: image_label, keep_prob: 1.0, global_step: epoch})
                accuracy_fp = acc_fp.eval(feed_dict={image: image_tensor, label: image_label, keep_prob: 1.0, global_step: epoch})
                lr = learning_rate.eval(feed_dict={image: image_tensor, label: image_label, keep_prob: 1.0, global_step: epoch})
                print (datetime.datetime.now())
                print ("Learning rate is %e"%(lr))
                print ("Loss is %g, detection accuracy is %g, IOU accuracy is %g, class accuracy is %g, false positive is %g"%(loss, accuracy_detect, accuracy_iou, accuracy_class, accuracy_fp))
            writer.add_summary(summary, epoch)
        writer.close()    
        coord.request_stop()
        coord.join(threads)
        
        save_path = saver.save(sess, "./tmp/My_Model.ckpt")
        print("Model saved in file: %s" % save_path)



            
def main():
    ne = 5000
    bs = 50
    net = Net(True, bs)
    solver = Solver(ne, bs, net)
    
    solver.train()
    
if __name__ == '__main__':
    main()
        