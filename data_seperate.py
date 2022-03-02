import random
from numpy import *
import cv2
import shutil
import os, sys



def split_data():
    print("deal with the data")
    if not os.path.exists('test/cnv'):
        os.mkdir('test/cnv')
    if not os.path.exists('test/pcv'):
        os.mkdir('test/pcv')
    if not os.path.exists('train/cnv'):
        os.mkdir('train/cnv')
    if not os.path.exists('train/pcv'):
        os.mkdir('train/pcv')

    shutil.rmtree('test/cnv')
    shutil.rmtree('test/pcv')
    shutil.rmtree('train/cnv')
    shutil.rmtree('train/pcv')

    os.mkdir('test/cnv')
    os.mkdir('test/pcv')
    os.mkdir('train/cnv')
    os.mkdir('train/pcv')

    cnv = arange(200) + 1
    pcv = arange(200) + 1
    random.shuffle(cnv)
    random.shuffle(pcv)

    cnv_test = cnv[0:20]
    pcv_test = pcv[0:20]

    cnv_train = cnv[20:200]
    pcv_train = pcv[20:200]

    for i in range(len(cnv_test)):
        img = cv2.imread('all_cnv/' + str(cnv_test[i]) + '.jpg')
        cv2.imwrite('test/cnv/' + str(cnv_test[i]) + '.jpg', img)

    for i in range(len(pcv_test)):
        img = cv2.imread('all_pcv/' + str(pcv_test[i]) + '.jpg')
        cv2.imwrite('test/pcv/' + str(pcv_test[i]) + '.jpg', img)

    for i in range(len(cnv_train)):
        img = cv2.imread('all_cnv/' + str(cnv_train[i]) + '.jpg')
        cv2.imwrite('train/cnv/' + str(cnv_train[i]) + '.jpg', img)

    for i in range(len(pcv_train)):
        img = cv2.imread('all_pcv/' + str(pcv_train[i]) + '.jpg')
        cv2.imwrite('train/pcv/' + str(pcv_train[i]) + '.jpg', img)


if __name__ == '__main__':
    split_data()