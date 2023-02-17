
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# 将输入图片归一化
def PlotImage(image):

    im = image.astype(float)
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def SRM(imgs):
    # 第一层滤波器
    # 定义三个滤波器,滤波器大小为5x5
    # filter1: egde3*3
    filter2 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    # filter2：egde5*5
    filter1 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    # filter3：一阶线性
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, -2, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    # 定义q，将三个滤波器归一化
    q = [4.0, 12.0, 2.0]
    filter1 = np.asarray(filter1, dtype=float) / 4
    filter2 = np.asarray(filter2, dtype=float) / 12
    filter3 = np.asarray(filter3, dtype=float) / 2
    # 将不同类的滤波器堆叠、处理，得到新滤波器
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]  # (3,3,5,5)

    filters = torch.FloatTensor(np.array(filters))  # (3,3,5,5)
    imgs = np.array(imgs, dtype=float)  # (375,500,3)
    imgs = np.transpose(imgs, (0,3, 1, 2))

    input = torch.tensor(imgs, dtype=torch.float32)




    # 定义第二层滤波器，滤波方式同第一层
    q = [4.0, 12.0, 2.0]
    filter2 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    # filter2：egde5*5
    filter1 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    # filter3：一阶线性
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, -2, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = torch.tensor(np.array(filters), dtype=torch.float32, requires_grad=False)


    def truncate_2(x):
        neg = ((x + 2) + abs(x + 2)) / 2 - 2
        return -(-neg + 2 + abs(- neg + 2)) / 2 + 2

    op2 = F.conv2d(input, filters, stride=1, padding=2)
    op2 = truncate_2(op2)



    op2 = op2[0]

    out = np.array(op2, dtype=float)


    return out


if __name__ == '__main__':

    img_cv = cv2.imread(r'D:\datasets\ct_columbia\4cam_splc\canong3_kodakdcs330_sub_03.tif')
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    noise_srm1 = SRM([img_cv])
    noise_srm1 = np.transpose(noise_srm1, (1, 2, 0))


    # plt.figure()
    # plt.imshow(noise_srm1.astype(np.uint8))
    # plt.show()
    image = cv2.cvtColor(noise_srm1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    print(image.shape)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.show()
    print(image)
