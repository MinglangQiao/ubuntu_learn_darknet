import cv2
import numpy as np
import sys
from math import radians, cos, sin, asin, sqrt, log
import math
from scipy.misc import imsave
import subprocess
from read_yuv import yuv_import
# from config import cluster_name, cluster_current
import PIL.Image as Image
from numpy import *
import os
from read_yuv import read_YUV420, merge_YUV2RGB_v1

def get_view(input_width,input_height,view_fov_x,view_fov_y,view_center_lat,view_center_lon,output_width,output_height,cur_frame, video_name, file_, is_save=False, save_dir=""):

    temp_1 = save_dir + video_name +'_frame_'+ str(cur_frame) +".yuv" # to save the converted yuv
    save_image_path = save_dir + video_name +'_frame_'+ str('%03d'%cur_frame) +".png"
    import config

    '''for more about the command, refere to this link:
        http://blog.csdn.net/sdlyjzh/article/details/8246752
       and you and type the ./remap(a binary file) in the terminal, ti should the she meaning of each command
    '''
    subprocess.call(["/home/ml/remap" , "-i" + "rect", "-o", "view", "-m", str(input_height), "-b", str(input_width), "-w", str(output_width), "-h", str(output_height), "-x",
                    str(view_fov_x), "-y",str(view_fov_y), "-p", str(view_center_lat), "-l", str(view_center_lon), "-z", "1", "-s", str(cur_frame), file_, temp_1])

                     # ./remap -i rect -o view -m 1920 -b 3840 -w 1080 -h 1200 -x 110 -y 113 -p 0 -l 0 -z 1 -s 10 '/media/ml/Data0/yuv/A380.yuv' '/home/ml/learn_darknet/src/frmae_1.yuv'
    try:
        Y, U, V = read_YUV420(temp_1, output_height, output_width)

        dst = merge_YUV2RGB_v1(Y, U, V)

        if is_save == True:
            cv2.imwrite(save_image_path, dst)
        else:
            cv2.imshow((video_name +'_frame_'+ str('%03d'%cur_frame) +".png"), dst)
            cv2.waitKey(0)

        subprocess.call(["rm", temp_1])
    except Exception as e:
        print('>>>>>>>>>>>>>exception: ', e)





    # frame = yuv_import(temp_1,(output_height,output_width),1,0)
    # print('>>>>>>>>>>>>> frame: ', frame)
    # subprocess.call(["rm", temp_1])

    # if(is_render==True):
    #
    #     print("this is debugging, not trainning")
    #     print(np.shape(np.array(frame)))
    #     YY=frame[0]
    #     im=Image.frombytes('L',(output_height,output_width),YY.tostring())
    #     im.show()
    #     frame = np.zeros((42,42,1))
    #     frame = np.reshape(frame, [42, 42, 1])
    #
    # else:
    #
    #     frame = np.array(frame)
    #     frame = frame.astype(np.float32)
    #     frame *= (1.0 / 255.0)
    #     frame = np.reshape(frame, [42, 42, 1])

    # return frame
