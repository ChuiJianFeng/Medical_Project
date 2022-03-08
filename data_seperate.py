import random
from numpy import *
import cv2
import shutil
import os, sys


def rename(name):

    filelist = os.listdir(name)
    total_num = len(filelist)

    i = 1

    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(name), item)
            if i < 2000:
                dst = os.path.join(os.path.abspath(name), str(i) + '.png')
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))
        elif item.endswith('.png'):
            src = os.path.join(os.path.abspath(name), item)
            if i < 2000:
                dst = os.path.join(os.path.abspath(name), str(i) + '.png')
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))
        i = i + 1

    print('total %d to rename & converted %d pngs' % (total_num, i))

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
    if not os.path.exists('val/cnv'):
        os.mkdir('val/cnv')
    if not os.path.exists('val/pcv'):
        os.mkdir('val/pcv')

    shutil.rmtree('test/cnv')
    shutil.rmtree('test/pcv')
    shutil.rmtree('train/cnv')
    shutil.rmtree('train/pcv')
    shutil.rmtree('val/cnv')
    shutil.rmtree('val/pcv')

    os.mkdir('test/cnv')
    os.mkdir('test/pcv')
    os.mkdir('train/cnv')
    os.mkdir('train/pcv')
    os.mkdir('val/cnv')
    os.mkdir('val/pcv')

    cnv = arange(896) + 1
    pcv = arange(822) + 1

    random.shuffle(cnv)
    random.shuffle(pcv)

    cnv_test = cnv[0:189]
    pcv_test = pcv[0:164]
    cnv_val= cnv[189:267]
    pcv_val = pcv[167:247]
    cnv_train = cnv[267:896]
    pcv_train = pcv[247:822]

    file = ['all_cnv', 'all_pcv']
    for i in range(2):
        rename(file[i])

    for i in range(len(cnv_test)):
        img = cv2.imread('all_cnv/' + str(cnv_test[i]) + '.png')
        cv2.imwrite('test/cnv/' + str(cnv_test[i]) + '.png', img)

    for i in range(len(pcv_test)):
        img = cv2.imread('all_pcv/' + str(pcv_test[i]) + '.png')
        cv2.imwrite('test/pcv/' + str(pcv_test[i]) + '.png', img)

    for i in range(len(cnv_val)):
        img = cv2.imread('all_cnv/' + str(cnv_val[i]) + '.png')
        cv2.imwrite('val/cnv/' + str(cnv_val[i]) + '.png', img)

    for i in range(len(pcv_val)):
        img = cv2.imread('all_pcv/' + str(pcv_val[i]) + '.png')
        cv2.imwrite('val/pcv/' + str(pcv_val[i]) + '.png', img)

    for i in range(len(cnv_train)):
        img = cv2.imread('all_cnv/' + str(cnv_train[i]) + '.png')
        cv2.imwrite('train/cnv/' + str(cnv_train[i]) + '.png', img)

    for i in range(len(pcv_train)):
        img = cv2.imread('all_pcv/' + str(pcv_train[i]) + '.png')
        cv2.imwrite('train/pcv/' + str(pcv_train[i]) + '.png', img)


if __name__ == '__main__':
    split_data()