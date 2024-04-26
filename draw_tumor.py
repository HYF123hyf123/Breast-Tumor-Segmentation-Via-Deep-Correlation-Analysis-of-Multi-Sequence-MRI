#!/usr/local/bin/python3
import cv2
import numpy as np
from PIL import Image
import pandas as pd

w = 256
h = 256
# Make empty black image
# image = cv2.imread("E:\\RIDERBreast\\P1\\5\\ear\\000019.png")
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# gray = Image.fromarray(image)
#将图片保存到当前路径下，参数为保存的文件名
# gray.save('gray.png')
# image = cv2.imread("gray.png")
# print(image)
image = cv2.imread("./stateOfArt_new/result/0_0ligege3175465_51/51_4.png")
image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA)  # 把原始背景图像进行缩小

# image = np.zeros((h,w,3),np.uint8)

# # Fill left half with yellow
# image[:,0:int(w/2)]=(0,255,255)
#
# # Fill right half with blue
# image[:,int(w/2):w]=(255,0,0)
#
# # Create a named colour
red = [0, 0, 255]
yellow = [0, 255, 255]
blue = [255, 0, 0]

# Change one pixel
# image[10,5]=red

####################################读取predict###############################
# predict = Image.open("E:\\RIDERBreast\\P1\\5\\mask\\000016.png").convert('L')
predict = Image.open("./stateOfArt_new/0zhe/99/save_img/hello53.jpg").convert('L')
# predict = Image.open("./stateOfArt_new/result/0_0ligege3175465_51/mask.png").convert('L')
# predict = Image.open("./stateOfArt_new/save_img/hello53.jpg").convert('L')
# predict = Image.open("./stateOfArt_new/2zhe/59/save_img/hello19.jpg").convert('L')
# predict = cv2.resize(predict, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
predict = predict.resize((256, 256), resample=0)  # 把预测结果进行放大
print("predict", predict)
# 用于表示改变图像过程用的差值方法。0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法。

predict_np = np.array(predict)
m = len(predict_np)
n = len(predict_np[0])
print("m", m)
print("n", n)
for i in range(m):
    for j in range(n):
        if predict_np[i][j] >= 34:
            print(predict_np[i][j])
            image[i, j] = red

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Save
cv2.imwrite("./stateOfArt_new/result/0_0ligege3175465_51/0zhe_99_hello53.png", image)
