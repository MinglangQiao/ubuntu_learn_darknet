import numpy as np
import subprocess
import os


#config
from config import  video_dic,  subject_dic, video_path, detect_threshold
from support import get_video_config


def get_frame_txt_rename(raw_image_path, frame, threshold):

    txt_path = '/home/ml/darknet/detection_box.txt'
    image_path = '/home/ml/darknet/predictions.png'

    os.chdir("/home/ml/darknet/")
    subprocess.call(["./darknet detect cfg/yolo.cfg yolo.weights " +  raw_image_path + " -thresh " + str(threshold)], shell=True)
    new_name_txt =  'test_' + '%03d'%frame + '.txt'
    new_name_image = 'test_'+ '%03d'%frame + '.png'

    find_flag = True
    find_flag_0 = False
    find_flag_1 = False
    ## rename txt and image
    while(find_flag):
        if os.path.exists(txt_path) is True:
            'read txt'
            os.rename("detection_box.txt",new_name_txt)
            find_flag_0 = True

        if os.path.exists(image_path) is True:
            find_flag_1 = True
            'read image'
            os.rename("predictions.png",new_name_image)

        if find_flag_0 == True and find_flag_1 == True:
            find_flag = False


def get_one_video_predictions():

    for i_video in range(len(video_dic)):
        if i_video == 70:

            'get video config'
            FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])

            for i_frame in range(1, FRAMESCOUNT + 1):
                if i_frame >= 1:

                    image_path = '/media/ml/Data1/2_0925_全景视频的眼动研究/Salience_of_FOV/程序/Finding_Content/' + video_dic[i_video] + '_raw_frames/'
                    read_image_path = image_path + '%03d'%i_frame + '.png'

                    get_frame_txt_rename(read_image_path, i_frame, detec_threshold[i_video])

                    print('>>>>>> processing: video_%d, %s, frame_%03d'%(i_video, video_dic[i_video], i_frame))


def get_frame_txt_rename_faster(raw_video_path, video_name, frame, threshold):

    os.chdir("/home/ml/darknet/")
    txt_path = '/home/ml/darknet/detection_box.txt'
    new_name_txt =  'test_' + '%03d'%frame + '.txt'

    find_flag = True

    ## rename txt and image
    while(find_flag):
        if os.path.exists(txt_path) is True:
            'read txt'
            os.rename("detection_box.txt",new_name_txt)
            find_flag = False

            'copy to target dict'
            save_path = 'results/' + video_name + '/'
            if os.path.exists(save_path) is False:
                os.mkdir(save_path)

            print(save_path)
            subprocess.call(["mv test_*.txt " + save_path], shell=True)
            subprocess.call(["mv output_*.png " + save_path], shell=True)

    return find_flag


def command_detect_video(raw_video_path, threshold):
    os.chdir("/home/ml/darknet/")
    subprocess.call(["./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights  -prefix output  " +  raw_video_path + " -thresh " + str(threshold)], shell=True)


def get_one_video_predictions_faster():

    for i_video in range(len(video_dic)):
        if i_video == 70:

            'get video config'
            video_path = '/media/ml/Data1/2_0925_全景视频的眼动研究/VR_杨燕丹/Video_All/'
            read_video_path = video_path + video_dic[i_video] + '.mp4'

            FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])
            'start the detect'
            # command_detect_video(read_video_path, detect_threshold[i_video])

            for i_frame in range(1, FRAMESCOUNT + 1):
                if i_frame >= 1:

                    find_flag = True

                    while(find_flag):
                        find_flag = get_frame_txt_rename_faster(read_video_path, video_dic[i_video], i_frame, detect_threshold[i_video])

                    print('>>>>>> processing: video_%d, %s, frame_%03d'%(i_video, video_dic[i_video], i_frame))





def run():
    print('hello!')
    get_one_video_predictions_faster() # then run the commnand in annother terminal























if __name__ == '__main__':
    run()
