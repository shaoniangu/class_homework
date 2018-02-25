# -*-coding:utf-8-*-
# __Author__=Youzhi Gu
# Learn Python at Zhejiang University


from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 原图像显示
img = cv2.imread('Campus scene of ZJU.jpg', 0)
cv2.imshow('Filtering Camparison', img)
cv2.imwrite('Row gray image.jpg',img)

# 均值滤波
blur_average = cv2.blur(img, (3, 5))    # 模板大小3*5
cv2.imshow('Average filtering', blur_average)
cv2.imwrite('Average filtering.jpg', blur_average)

# 高斯滤波
img2 = img  # 复制一个img，防止原图像变量img被损坏
for i in range(2000):   # 添加点噪声
    temp_x = np.random.randint(0, img2.shape[0])
    temp_y = np.random.randint(0, img2.shape[1])
    img2[temp_x][temp_y] = 255
cv2.imshow('Add noise to img',img2)
cv2.imwrite('Add noise to image.jpg', img2)
blur_Gaussian = cv2.GaussianBlur(img2, (5, 5), 0)
cv2.imshow('Gaussian filtering', blur_Gaussian)
cv2.imwrite('Gaussian filtering.jpg', blur_Gaussian)

# 中值滤波
blur_median = cv2.medianBlur(img2, 5)
cv2.imshow('Median filtering', blur_median)
cv2.imwrite('Median filtering.jpg', blur_median)

# 双边滤波
blur_bilateral = cv2.bilateralFilter(img2, 9, 100, 100)
cv2.imshow('Bilateral filtering', blur_bilateral)
cv2.imwrite('Bilateral filtering.jpg', blur_bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
