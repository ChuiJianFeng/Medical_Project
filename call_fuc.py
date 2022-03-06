from data_seperate import split_data
from test import test_data
import os
import numpy as np
from torchinfo import summary
# fileDir = r"D:\Jian\110-下學期\醫療影像專題\MedicalImage_Project01_Classification"
# fileExt = r".pth"

file= 'model_name.txt'


if __name__ == '__main__':
    i = 0
    '''
    while i<=5:
        split_data()
        os.system("python train.py")

        i +=1

    i = 0
    while i<=5:
        with open(file, 'r') as f:
            PATH_TO_WEIGHTS = f.read().splitlines()
        tmp = test_data(PATH_TO_WEIGHTS[i])
        tmp.test()
        i += 1
    '''
    with open('ACC.txt', 'r') as f2:
        A = []
        acc = f2.read()
        acc = [line for line in acc.split("\n")]
        for i in range(len(acc)):
             A.append(float(acc[i]))
    print("ACC : ",np.average(A))
