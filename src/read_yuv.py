# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:48:00 2013

@author: Chen Ming
"""

from numpy import *
import numpy as np
import cv2

screenLevels = 255.0


def yuv_import(filename,dims,numfrm,startfrm):
    fp=open(filename,'rb')
    blk_size = prod(dims) * 1
    fp.seek(blk_size*startfrm,0)
    Y=[]
    # U=[]
    # V=[]
    # print dims[0]
    # print dims[1]
    # d00=dims[0]//2
    # d01=dims[1]//2
    # print d00
    # print d01
    Yt=zeros((dims[0],dims[1]),uint8,'C')
    # Ut=zeros((d00,d01),uint8,'C')
    # Vt=zeros((d00,d01),uint8,'C')
    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                #print m,n
                Yt[m,n]=ord(fp.read(1))
        # for m in range(d00):
        #     for n in range(d01):
        #         Ut[m,n]=ord(fp.read(1))
        # for m in range(d00):
        #     for n in range(d01):
        #         Vt[m,n]=ord(fp.read(1))
        Y=Y+[Yt]
        # U=U+[Ut]
        # V=V+[Vt]
    fp.close()
    return Y


def yuv_import_3(filename,dims,numfrm,startfrm):
    """
    read 420p nv12
    """
    preview = True

    fp=open(filename,'rb') # read in binary
    blk_size = int(prod(dims) * 3/2) # get total lenth of the yuv
    fp.seek(blk_size*startfrm,0)
    Y=[]
    U=[]
    V=[]
    d00=dims[0]//2 # 表示整数除法，返回不大于结果的一个最大的整数
    d01=dims[1]//2
    Yt=zeros((dims[0],dims[1]),uint8,'C')
    Ut=zeros((d00,d01),uint8,'C')
    Vt=zeros((d00,d01),uint8,'C')

    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                #print m,n
                Yt[m,n]=ord(fp.read(1)) # fp的指针应该是读一次就往后移动了

        if not preview: #capture nv12
            for m in range(d00):
                for n in range(d01):
                    Ut[m,n]=ord(fp.read(1))
                    Vt[m,n]=ord(fp.read(1))
        else:#preview nv21
            for m in range(d00):
                for n in range(d01):
                    Vt[m,n]=ord(fp.read(1))
                    Ut[m,n]=ord(fp.read(1))

        # for m in range(d00):
        #     for n in range(d01):
        #         Vt[m,n]=ord(fp.read(1))
        # for m in range(d00):
        #     for n in range(d01):
        #         Ut[m,n]=ord(fp.read(1))
        Y=Y+[Yt]
        U=U+[Ut]
        V=V+[Vt]
    fp.close()
    return (Y,U,V)

'works for nv21, not yv/yu 420'
def yuv2rgb(Y,U,V,width,height):
    U=repeat(U,2,0)
    U=repeat(U,2,1)
    V=repeat(V,2,0)
    V=repeat(V,2,1)
    rr=zeros((width,height),float,'C')
    gg=zeros((width,height),float,'C')
    bb=zeros((width,height),float,'C')

    Y = Y.astype(float)
    U = U.astype(float)
    V = V.astype(float)


    bb = 1.164 * (Y-16) + 2.018 * (U - 128)
    gg = 1.164 * (Y-16) - 0.813 * (V - 128) - 0.391 * (U - 128)
    rr = 1.164 * (Y-16) + 1.596*(V - 128)

    # rr= Y+1.14*(V-128.0)
    # gg= Y-0.395*(U-128.0)-0.581*(V-128.0)
    # bb= Y+2.032*(U-128.0)             # 必须是128.0，否则出错

    # rr = Y + (1.370705 * (V-128.0));
    # gg = Y - (0.698001 * (V-128.0)) - (0.337633 * (U-128.0));
    # bb = Y + (1.732446 * (V-128.0));

    rr = clip(rr, 0, 255)
    gg = clip(gg, 0, 255)
    bb = clip(bb, 0, 255)


    rr1=rr.astype(uint8)
    gg1=gg.astype(uint8)
    bb1=bb.astype(uint8)


    return rr1,gg1,bb1

'the final version, read only one image'
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
    # print(type(gray))
    # print(gray.shape)

    # create U,V
    img_U = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(img_U))
    # print(img_U.shape)

    img_V = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(img_V))
    # print(img_V.shape)

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

'works for yuv420p'
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


# if __name__ == '__main__':
#     data=yuv_import('E:\\new\\test\\ballroom\\ballroom_0.yuv',(480,640),1,0)
#     #print data
#     #im=array2image(array(data[0][0]))
#     YY=data[0][0]
#     print YY.shape
#     for m in range(2):
#         print m,': ', YY[m,:]
#
#     im=Image.fromstring('L',(640,480),YY.tostring())
#     im.show()
#     im.save('f:\\a.jpg')
