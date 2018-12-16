import keras, cv2, json, time, os
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input,AveragePooling2D
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing, metrics
from tqdm import tqdm
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import skimage.measure as ski
import keras.backend.tensorflow_backend as TF
import tensorflow as tf



# 变量定义
#############################################
classes = to_categorical([i for i in range(52)])
chars = [chr(i) for i in range(ord('a'),ord('z')+1)]+[chr(i) for i in range(ord('A'),ord('Z')+1)]


# 工具类函数(CPU)
#############################################
def a2c(array_list):
    "传入四个字符的预测结果的独热编码"
    buffer = ''
    for code in array_list:
        buffer += chars[list(code).index(max(code))]
    return buffer

def removeContour(img_t, inter=25):
    """基于连通域去噪"""
    lbs = ski.label(img_t,connectivity=1,background=False)
    flat = list(lbs.flatten())
    uni = np.unique(lbs)
    for e in uni:
        count = flat.count(e)
        if  count < inter or e == 0:
            img_t[lbs==e] = 0
        else:
            img_t[lbs==e] = 1 #前景设为1
    return img_t

def labels(path = './file/train_list.json'):
    with open(path, 'r') as f:
        labels = json.load(f)
        return labels


def preParse(img, cut=True):
    """传入一个img对象,预处理图像"""
    width = 40
    # 模糊处理
    # img = cv2.blur(img,(3,3))
    # img = cv2.blur(img, (2, 2))
    # ret, img = cv2.threshold(img[3:43], 155, 1, cv2.THRESH_OTSU)
    # ret, img = cv2.threshold(img[3:43], 155, 1, cv2.THRESH_BINARY)
    # img = 1 - img # 背景为0
    img = img[2:42]/255
    img = np.reshape(img,(40,-1,1))
    if cut:
        return img[:, 5:5 + width], \
               img[:, 60:60 + width], \
                img[:, 120:120 + width], \
                img[:, 180:180 + width]
    return img
           

def feed(labels):
    X, Y = [], []
    for sample in tqdm(labels,ascii=True,ncols=50):
        X += preParse(cv2.imread(sample[0],0))
        Y += [c for c in sample[1]]
    Y = [classes[chars.index(i)] for i in Y]
    return np.array(X), np.array(Y)

def feed_muti(labels):
    X, Y = [], []
    for sample in tqdm(labels,ascii=True,ncols=50):
        X.append(preParse(cv2.imread(sample[0],0),cut=False))
        Y.append([classes[chars.index(c)] for c in sample[1]])
    X,Y = np.array(X), np.array(Y)
    print(X.shape,Y.shape)
    return X,Y


if __name__ == '__main__':
        print(pow(0.91,4))