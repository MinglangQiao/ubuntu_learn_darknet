# coding=UTF-8
import numpy as np
import subprocess
import os
from multiprocessing import Process, Queue
# from vrplayer import get_view


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

                    print('>>>>>> processing: video_%d, %s, frame_%03d/%03d'%(i_video, video_dic[i_video], i_frame, FRAMESCOUNT))


def get_frame_txt_rename_faster(raw_video_path, video_name, frame, threshold):

    os.chdir("/home/ml/darknet/")
    txt_path = '/home/ml/darknet/detection_box.txt'
    new_name_txt =  'test_' + '%03d'%frame + '.txt'

    find_flag = True

    ## rename txt and image
    'copy to target dict'
    save_path = '/home/ml/Data/get_box/' + video_name + '/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    while(find_flag):
        if os.path.exists(txt_path) is True:
            'read txt'
            os.rename("detection_box.txt",new_name_txt)
            find_flag = False

            # print(save_path)
            # subprocess.call(["mv test_*.txt " + save_path], shell=True)
            # subprocess.call(["mv output_*.png " + save_path], shell=True)
            subprocess.call(["mv test_*.txt " + save_path], shell=True)
            subprocess.call(["mv output_*.jpg " + save_path], shell=True)

    return find_flag


def command_detect_video(raw_video_path, threshold):
    os.chdir("/home/ml/darknet/")
    subprocess.call(["./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights  -prefix output " + raw_video_path + " -thresh " + str(threshold)], shell = True)

def command_detect_multi_image():
    os.chdir("/home/ml/darknet/")
    subprocess.call(["./darknet detect cfg/yolo.cfg yolo.weights"], shell = True)

def get_terminal_out():
    os.chdir("/home/ml/darknet/")
    pipe = subprocess.Popen("m", shell=True, stdout=PIPE).stdout
    output = pipe.read()

    print('>>>>>>>>>>>>>>get the output', output)

def command_print_iamge_path(im_path):
    os.chdir("/home/ml/darknet/")
    subprocess.call([im_path], shell = True)


def get_one_video_predictions_faster(i_video):

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

def get_fov_image():
    pass

def get_yuv(video_path, output_path):

    subprocess.call(["ffmpeg -i " + video_path + " -c:v rawvideo -pix_fmt yuv420p " + output_path], shell = True)


def run():
    print('hello!')

    'config'
    # ----------------------------------------------
    mode_dic = {
        1: 'get_prediction_box',
        2: 'get_fov',
        3: 'get_yuv',
        4: 'test_yuv_import', # just for debug
        5: 'get_fov_object'
    }

    mode  = mode_dic[ 1 ]

    if mode == 'get_prediction_box':
            import cv2
            import subprocess

            sub_mode = 'get_box' # 'get_box', 'get_sta', 'get_box_multi_frames'

            if sub_mode == 'get_box':

                for i_video in range(len(video_dic)):
                    if i_video == 0:

                        video_path = '/home/ml/Data/Video_All/'
                        read_video_path = video_path + video_dic[i_video] + '.mp4'

                        # subprocess.call(["./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights  -prefix output /media/ml/Data1/2_0925_全景视频的眼动研究/VR_杨燕丹/Video_All/VRBasketball.mp4 -thresh 0.6"], shell = True)

                        p_rename = Process(target = get_one_video_predictions_faster, args = (i_video, ))
                        # print('>>>>>>>>>>>.d1: ', read_video_path, detect_threshold[i_video])

                        p_video_detection = Process(target = command_detect_video, args = (read_video_path, detect_threshold[i_video]))

                        p_rename.start()
                        p_video_detection.start()

                        p_video_detection.join()
                        p_rename.join()

                        p_rename.terminate()

            if sub_mode == 'get_box_multi_frames':
                """
                not finish yet
                """
                os.chdir("/home/ml/darknet/")

                for i_video in range(len(video_dic)):
                    if i_video == 70:

                        from config import video_path
                        from read_yuv import read_YUV420, merge_YUV2RGB_v1
                        import subprocess

                        'config'
                        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])

                        video_path = '/home/ml/Data/Video_All/'
                        read_video_path = video_path + video_dic[i_video] + '.mp4'

                        'start detect:'
                        p_rename = Process(target = get_one_video_predictions_faster, args = (i_video, ))
                        p_rename.start()

                        # p_multi_frame = Process(target = command_detect_multi_image, args = ())

                        p = subprocess.Popen(["./darknet", "detect", "cfg/yolo.cfg", "yolo.weights",
                        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        for i_frame in range(1,FRAMESCOUNT + 1):
                            if i_frame == 1:

                                image_path = '/media/ml/Data1/2_0925_全景视频的眼动研究/Salience_of_FOV/程序/Finding_Content/A380_raw_frames/'
                                read_video_path = image_path + '%03d'%i_frame + '.png'
                                print('>>>>>>>>>>>>>>>>>>>. get here')

                                # p_multi_frame.start()

                                output, err = p.communicate(b'set data/dog.jpg\n')
                                print('>>>>>>>>>>>>output: ', output.decode('utf-8'))

                                # get_terminal_out = Process(target = get_terminal_out, args = ())

                                # get_terminal_out.start()

                                # p_video_detection.start()

                                # p_video_detection.join()
                                # p_rename.join()

                                # p_rename.terminate()

            if sub_mode ==  'get_sta':
                """
                statistic the proportion of fixations fall into the box
                """
                from config import video_path
                from read_yuv import read_YUV420, merge_YUV2RGB_v1
                import subprocess

                for i_video in range(len(video_dic)):
                    if  i_video == 70:
                        'config'
                        FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])

                        for i_frame in range(1, FRAMESCOUNT + 1):
                            if  i_frame == 1:
                                read_box_path = '/home/ml/darknet/results/' + video_dic[i_video] + '/test_' + '%03d'%i_video + '.txt'

                                f = open(read_box_path)
                                lines = f.readlines()
                                temp_file = []

                                for line in lines:
                                    line = line.split()
                                    line = [i for i in line]
                                    temp_file.append(line)

                                print(temp_file)


    if mode == 'get_yuv':

        import subprocess
        for i_video in range(len(video_dic)):
            if i_video == 37:
                video_path = '/media/ml/Data1/2_0925_全景视频的眼动研究/VR_杨燕丹/Video_All/'
                read_video_path = video_path + video_dic[i_video] + '.mp4'

                out_put_path = '/media/ml/Data0/yuv/'
                yuv_path = out_put_path + video_dic[i_video] + '.yuv'

                get_yuv(read_video_path, yuv_path)

    if mode == 'get_fov':
        print('>>>>>>>>>>>>>>>>>>>>>>>>mode change to: %s.' % mode)

        from vrplayer import get_view
        from config import video_path
        from read_yuv import read_YUV420, merge_YUV2RGB_v1
        import cv2
        import subprocess

        for i_video in range(len(video_dic)):
            if  i_video == 70:
                'config'
                FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])
                'config part1'
                video_size_width = IMAGEWIDTH
                video_size_heigth = IMAGEHEIGHT
                output_height = 1200
                output_width = 1080

                view_range_lon = 110
                view_range_lat = 113

                for i_frame in range(1, FRAMESCOUNT + 1):
                    if  i_frame == 400:
                        'do not need png'
                        # image_path = '/media/ml/Data1/2_0925_全景视频的眼动研究/Salience_of_FOV/程序/Finding_Content/' + video_dic[i_video] + '_raw_frames/'
                        # read_image_path = image_path + '%03d'%i_frame + '.png'

                        'config part2'
                        cur_lon = 90
                        cur_lat = 0
                        temp_dir = "test_object/"

                        '''
                        the input is video/frame yuv, the out put is also yuv
                        for more about yuv and the meaning of the command, refere this link:
                            http://blog.csdn.net/beyond_cn/article/details/12998247
                        '''
                        cur_observation = get_view(input_width=video_size_width,
                                                   input_height=video_size_heigth,
                                                   view_fov_x=view_range_lon,
                                                   view_fov_y=view_range_lat,
                                                   cur_frame=i_frame,
                                                   is_save=False,
                                                   output_width=output_width,
                                                   output_height=output_height,
                                                   view_center_lon=cur_lon,
                                                   view_center_lat=cur_lat,
                                                   video_name = video_dic[i_video],
                                                   save_dir=temp_dir,
                                                   file_='/media/ml/Data0/yuv/' + video_dic[i_video] + '.yuv')

    if mode == 'test_yuv_import':
        print('>>>>>>>>>>>>>>>>>>>>>>>>mode change to: %s.' % mode)
        from read_yuv import yuv_import
        import cv2
        from vrplayer import get_view
        from config import video_path
        from PIL import Image

        sub_mode  = 'test_yuv2' # 'convert', 'get_yuv_image', 'test_yuv2'(works)

        if sub_mode == 'convert':

            for i_video in range(len(video_dic)):
                if i_video == 0:
                    'config'
                    FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])

                    for i_frame in range(1, FRAMESCOUNT + 1):
                        if i_frame == 2:


                            yuv_file ='/media/ml/Data0/yuv/' + video_dic[i_video] + '.yuv'

                            print('>>>>>>: ', video_dic[i_video])
                            data = yuv_import(yuv_file, (IMAGEWIDTH, IMAGEHEIGHT), 1, 0)
                            data = np.array(data)
                            print(data)
                            y = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR) # convert to RGB
                            result = np.vstack([y])
                            cv2.imwrite('test_yuv.png', result)
                            print(data)

        if sub_mode == 'get_yuv_image':
             pass
             # img_in = cv2.imread('006.png')
             #
             # img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
             # print(img_out)


        if sub_mode == 'test_yuv':
                'have some problem'
                from read_yuv import yuv_import_3, yuv2rgb
                from PIL import Image

                temp_1 = 'frame_100.yuv'
                output_height = 1200
                output_width = 1080
                size = (output_width, output_height)
                data = yuv_import_3(temp_1,(output_height,output_width),1,0)
                print('>>>>>>>>>>>>>>>>> data: ', data)

                R_=data[0][0]
                G_=data[1][0]
                B_=data[2][0]
                RGB=yuv2rgb(R_,G_,B_,size[0],size[1])
                im_r=Image.frombytes('L',size,RGB[0].tostring())
                im_g=Image.frombytes('L',size,RGB[1].tostring())
                im_b=Image.frombytes('L',size,RGB[2].tostring())
                # im_r.show()
                # for m in range(2):
                #     print m,': ', R_[m,:]
                co=Image.merge('RGB', (im_r,im_g,im_b))
                # co.show()
                savePath = 'tett_yuv.png'
                print(savePath)
                co.save(savePath)

        if sub_mode == 'test_yuv2':
            'works for yuv 420P'
            from read_yuv import read_YUV420, merge_YUV2RGB_v1

            image_path = 'frame_100.yuv'
            output_height = 1200
            output_width = 1080

            Y, U, V = read_YUV420(image_path, output_height, output_width)

            dst = merge_YUV2RGB_v1(Y, U, V)

            cv2.imshow("dst", dst)
            cv2.imwrite('test_yuv.png', dst)
            cv2.waitKey(0) # not enter key in the terminal


    if mode == 'get_fov_object':
        print('>>>>>>>>>>>>>>>>>>>>>>>>mode change to: %s.' % mode)
        from config import subject_dic, video_path
        from support import get_raw_data, fov_detection, fov_center
        from vrplayer import get_view

        source_path = 'filtered_Data'
        for i_video in range(len(video_dic)):
            if  i_video == 70:

                'config'
                FRAMERATE, FRAMESCOUNT, IMAGEWIDTH, IMAGEHEIGHT = get_video_config(video_path, video_dic[i_video])
                leas_num = round(FRAMERATE * 2)

                'config part1: for get fov image'
                video_size_width = IMAGEWIDTH
                video_size_heigth = IMAGEHEIGHT
                output_height = 1200
                output_width = 1080

                view_range_lon = 110
                view_range_lat = 113


                'get the raw data'
                one_video_lon, one_video_lat, one_video_eye_x, one_video_eye_y, one_video_eye_lon, one_video_eye_lat = get_raw_data(
                    subject_dic, source_path, video_dic[i_video])
                # print('>>>>>>>>>>>>>>>: ',np.shape(one_video_lon))

                'detect the fov for each subject'
                one_video_all_subjects_fov_zone = []
                one_video_all_subjects_fov_center = []
                subplot_mode = 'no_detection_fov' # 'detect_fov', 'no_detection_fov'

                for i_subject in range(len(subject_dic)):
                    if  i_subject == 14:

                        if subplot_mode == 'detect_fov':
                            '>>>>>>>>>>>>>>>>>.step1: detecte the fov zone'
                            print('>>>>>>>>>>>>> video:%s_%s subject:%s_%s' % (video_dic[i_video], str(i_video), subject_dic[i_subject], str(i_subject)))

                            fov_zone = fov_detection(one_video_lon[i_subject][:], one_video_lat[i_subject][:], 5, leas_num)
                            print('>>>>>>>>>>>>>>>debug1: ',fov_zone)

                            fov_center_location = (fov_center(fov_zone, one_video_lon[i_subject][:], one_video_lat[i_subject][:]))
                            fov_center_location = np.round(fov_center_location)
                            print('>>>>>>>>>>>>>>>debug2: ', np.round(fov_center_location))

                            one_video_all_subjects_fov_zone.append(fov_zone)
                            one_video_all_subjects_fov_center.append(fov_center_location)
                            # print('>>>>>>>>>>>>>>>:debug3 ', one_video_all_subjects_fov_zone)
                            # print('>>>>>>>>>>>>>>>:debug3.1 ', one_video_all_subjects_fov_zone)

                            '>>>>>>>>>>>>>>>>>.step2: detecte the fov image'
                            save_fov_image_dir = "test_object/"
                            'for each fov zone'
                            for i_fov in range(len(fov_center_location)):

                                if i_fov >= 0:
                                    # cur_lon = fov_center_location[i_fov][0]
                                    # cur_lat = fov_center_location[i_fov][1]

                                    # i_frame = fov_zone[i_fov][0]

                                    'for each fov frame'
                                    for i in range(fov_zone[i_fov][0], fov_zone[i_fov][1]):
                                        i_frame = i
                                        cur_lon = one_video_lon[i_subject][i]
                                        cur_lat = one_video_lat[i_subject][i]

                                        cur_observation = get_view(input_width=video_size_width,
                                                                   input_height=video_size_heigth,
                                                                   view_fov_x=view_range_lon,
                                                                   view_fov_y=view_range_lat,
                                                                   cur_frame=i_frame,
                                                                   is_save=True,
                                                                   output_width=output_width,
                                                                   output_height=output_height,
                                                                   view_center_lon=cur_lon,
                                                                   view_center_lat=cur_lat,
                                                                   video_name = video_dic[i_video] + '_subject_' + str(i_subject) + '_fov_' + str(i_fov),
                                                                   save_dir=save_fov_image_dir,
                                                                   file_='/media/ml/Data0/yuv/' + video_dic[i_video]  + '.yuv')

                        if subplot_mode == 'no_detection_fov':
                            import subprocess

                            # save_fov_image_dir = "test_object/"
                            # for i in range(FRAMESCOUNT):
                            #
                            #     i_frame = i
                            #     cur_lon = one_video_lon[i_subject][i]
                            #     cur_lat = one_video_lat[i_subject][i]
                            #
                            #     cur_observation = get_view(input_width=video_size_width,
                            #                                input_height=video_size_heigth,
                            #                                view_fov_x=view_range_lon,
                            #                                view_fov_y=view_range_lat,
                            #                                cur_frame=i_frame,
                            #                                is_save=True,
                            #                                output_width=output_width,
                            #                                output_height=output_height,
                            #                                view_center_lon=cur_lon,
                            #                                view_center_lat=cur_lat,
                            #                                video_name = video_dic[i_video] + '_subject_' + str(i_subject),
                            #                                save_dir=save_fov_image_dir,
                            #                                file_='/media/ml/Data0/yuv/' + video_dic[i_video]  + '.yuv')

                            subprocess.call(['ffmpeg', '-r', str(FRAMERATE), '-i', 'test_object/VRBasketball_subject_1_frame_'+ '%03d' + '.png',
                                         '-b', '4000k', '-codec', 'mpeg4', 'test_object/'+ video_dic[i_video] + 'subject_' + str(i_subject) + '_fov.mp4'])

                            # subprocess.call(['rm *.png'])















if __name__ == '__main__':
    run()
