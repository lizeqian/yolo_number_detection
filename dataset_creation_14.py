import numpy as np
import os
import cv2
import random
import glob
from datetime import datetime

def overlap(x,y,w,h, start_x, start_y, width, height):
    num = len(start_x)
    overlap = False
    for i in range(num):
        x0 = start_x[i]
        y0 = start_y[i]
        x1 = x0+width[i]
        y1 = y0+height[i]
        lu_x = max(x0, x)
        lu_y = max(y0, y)
        rb_x = min(x1, x+w)
        rb_y = min(y1, y+h)

        if rb_x > lu_x and rb_y > lu_y:
            overlap = True
    return overlap

def crop_image(img):
    mean_x = np.mean(img, axis= 1)
    for i in range(len(mean_x)):
        if mean_x[i] > 1:
            start_point = i
            break

    for i in range(len(mean_x)):
        if mean_x[len(mean_x)-1-i] > 0:
            end_point = len(mean_x)-1-i
            break

    img = np.array(img)[:, start_point:end_point]
    return img

def random_placement(addrs, pic_w, pic_h, num_classes):
    img_ext = np.zeros((pic_h,pic_w), dtype=np.uint8)
    start_x=[]
    start_y=[]
    width = []
    height = []
    labels=[]
#    reX, reY = random.uniform(0.5,1.5),random.uniform(0.5,1.5)
    for i in range(10):
        y = random.randint(0, pic_h - 28*2 - 1)
        for j in range(10):
            label = random.randint(0, num_classes-1)
            img=cv2.imread(random.choice(addrs[label][0]),cv2.IMREAD_GRAYSCALE)
            reX, reY = random.uniform(1.5,2),random.uniform(1.5,2)
            img=cv2.resize(img,None,fx=reX, fy=reY, interpolation = cv2.INTER_CUBIC)
            img=crop_image(img)
            h = np.shape(img)[0]
            w = np.shape(img)[1]
            if j == 0:
                x = random.randint(0, pic_w - w - 1)
            if overlap(x,y,w,h, start_x, start_y, width, height):
                break
            else:
                if x + w < pic_w:
                    labels.append(label)
                    height.append(h)
                    width.append(w)
                    start_x.append(x)
                    start_y.append(y)
                    img_ext[y:y+h,x:x+w] = img

                    x+=w
                else:
                    break
    mask = np.where(img_ext==0)
    mask_val = np.random.randint(low=0, high=20, size=(pic_h, pic_w), dtype = np.uint8)
    img_ext[mask[0],mask[1]]=mask_val[mask[0],mask[1]]
    return img_ext, start_x, start_y, width, height, labels

if __name__ == '__main__':

    dataset_name = 'train14'
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    random.seed(datetime.now())
    temp_path = []
    cell_num = 14
    img_size = 448
    cell_w = img_size/cell_num

    num_classes = 13

    for i in range(num_classes):
#        import os
#        relevant_path = "./trainingSet/"
#        included_extensions = ['jpg', 'bmp', 'png', 'gif', 'JPG', 'PNG']
#        file_names = [fn for fn in os.listdir(relevant_path) if any(fn.endswith(ext) for ext in included_extensions)]
        temp_str = './trainingSet/'+str(i)+'/*.jpg'
        temp_path.append(temp_str)

    num_training = 10000
    num_val = 0
    num_test = 0

    addrs = []


    for i in range(num_classes):
        addrs.append([])
        addrs[i].append(glob.glob(temp_path[i]))


    csvarray = np.zeros((cell_num*cell_num*num_training, num_classes+5))

    for i in range(num_training):
        if i%100 == 0:
            print (datetime.now())
            print ("Gen training data %g"%(i))

        img = []
        resize = []
        img_resize = []
        size = []
        img, start_x, start_y, width, height, labels = random_placement(addrs, img_size, img_size, num_classes)
        img_savdir = './'+dataset_name+'/'+str(i)+'.jpg'
        cv2.imwrite(img_savdir,img)
        num_img = len(start_x)

        for j in range(num_img):
            x = start_x[j]+width[j]/2.0
            y = start_y[j]+height[j]/2.0
            w = width[j]
            h = height[j]
            label = labels[j]
            x_int = int(x//cell_w)
            y_int = int(y//cell_w)
            x_frac = (x%cell_w)/cell_w
            y_frac = (y%cell_w)/cell_w
            w_frac = w/img_size
            h_frac = h/img_size

            label_onehot = np.zeros(num_classes)
            label_onehot[label] = 1
            csvarray[i*cell_num*cell_num+y_int*cell_num+x_int, :4] = [x_frac, y_frac, w_frac, h_frac]
            csvarray[i*cell_num*cell_num+y_int*cell_num+x_int, 4] = 1
            csvarray[i*cell_num*cell_num+y_int*cell_num+x_int, 5:] = label_onehot
    np.savetxt(dataset_name+".csv", csvarray, fmt='%g', delimiter=',')