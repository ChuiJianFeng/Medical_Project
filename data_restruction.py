import cv2
import os
import numpy as np


def read_directory(directory_name):
    array_of_img = []
    files = []
    for filename in os.listdir(r"./" + directory_name):
        files.append(str(filename[0:-4]))
        img = cv2.imread(directory_name + "/" + filename)
        img_r1, img_r2, img_r3, flipVertical, flipHorizontal = rotate_img(img)
        array_of_img.append([img_r1, img_r2, img_r3, flipVertical, flipHorizontal])
    print(len(array_of_img))
    for i in range(len(array_of_img)):
        for j in range(len(array_of_img[0])):
            cv2.imwrite(directory_name + '/' + str(files[i]) + "_" + str(j) + '.jpg', array_of_img[i][j])
    print(len(array_of_img[0]))


def rotate_img(img):
    (h, w, d) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotate_img1 = cv2.warpAffine(img, M, (w, h))

    M1 = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotate_img2 = cv2.warpAffine(img, M1, (w, h))

    M2 = cv2.getRotationMatrix2D(center, 270, 1.0)
    rotate_img3 = cv2.warpAffine(img, M2, (w, h))

    flipVertical = cv2.flip(img, 0)
    flipHorizontal = cv2.flip(img, 1)
    flipBoth = cv2.flip(img, -1)

    return rotate_img1, rotate_img2, rotate_img3, flipVertical, flipHorizontal


if __name__ == '__main__':
    read_directory('test/cnv')
    read_directory('test/pcv')


