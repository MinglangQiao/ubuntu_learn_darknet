# -*- coding: utf-8 -*-

import cv2
import numpy as np


def read_YUV420(image_path, rows, cols):
    """
    读取YUV文件，解析为Y, U, V图像
    :param image_path: YUV图像路径
    :param rows: 给定高
    :param cols: 给定宽
    :return: 列表，[Y, U, V]
    """
    # create Y
    gray = np.zeros((rows, cols), np.uint8)
    print(type(gray))
    print(gray.shape)

    # create U,V
    img_U = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    print(type(img_U))
    print(img_U.shape)

    img_V = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    print(type(img_V))
    print(img_V.shape)

    with open(image_path, 'rb') as reader:
        for i in range(rows):
            for j in range(cols):
                gray[i, j] = ord(reader.read(1))

        for i in range(int(rows / 2)):
            for j in range(int(cols / 2)):
                img_U[i, j] = ord(reader.read(1))

        for i in range(int(rows / 2)):
            for j in range(int(cols / 2)):
                img_V[i, j] = ord(reader.read(1))

    return [gray, img_U, img_V]


def merge_YUV2RGB_v1(Y, U, V):
    """
    转换YUV图像为RGB格式（放大U、V）
    :param Y: Y分量图像
    :param U: U分量图像
    :param V: V分量图像
    :return: RGB格式图像
    """
    # Y分量图像比U、V分量图像大一倍，想要合并3个分量，需要先放大U、V分量和Y分量一样大小
    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 合并YUV3通道
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return dst

def merge_YUV2RGB_v2(Y, U, V):
    """
    转换YUV图像为RGB格式（缩小Y）
    :param Y: Y分量图像
    :param U: U分量图像
    :param V: V分量图像
    :return: RGB格式图像
    """
    rows, cols = Y.shape[:2]

    # 先缩小Y分量，合并3通道，转换为RGB格式图像后，再放大至原来大小
    shrink_Y = cv2.resize(Y, (int(cols / 2), int(rows / 2)), interpolation=cv2.INTER_AREA)

    # 合并YUV3通道
    img_YUV = cv2.merge([shrink_Y, U, V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    cv2.COLOR_YUV2BGR_I420

    # 放大
    enlarge_dst = cv2.resize(dst, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC) # INTER_AREA, INTER_CUBIC, INTER_NEAREST, INTER_LANCZOS4
    return enlarge_dst


if __name__ == '__main__':
    rows = 1200
    cols = 1080
    image_path = 'frame_100.yuv'

    Y, U, V = read_YUV420(image_path, rows, cols)

    dst = merge_YUV2RGB_v1(Y, U, V)

    cv2.imshow("dst", dst)
    cv2.imwrite('test_yuv.png', dst)
    cv2.waitKey(0) # not enter key in the terminal
