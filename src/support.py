import numpy as np
import imageio
from math import sin, cos, acos, radians
import math
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, log

def fov_detection(lon, lat, R_limit_set, length_val_set):
    '''detect one subject's fov'''
    # detection config
    # length_val = length_val_set  # 45 as the framerate is about 30 fps, one fov need at least 2s, the number of fixation in one fov
    # R_limit_= R_limit_set  # 3 degree

    fov_zone = []
    length_raw = length_val_set # the fixation number of a fov
    start_new = 0
    flag = 'not detect' # flag is the fixation is in fov

    lon = np.float32(lon)
    lat = np.float32(lat)

    while (start_new + length_raw - 1) < (len(lon)):
        start_point = start_new
        length_val = length_raw
        zone_delat_val = [] # save the delta value, remember to reset it

        center_lon = np.mean(lon[start_point : (start_point + length_val)])
        center_lat = np.mean(lat[start_point : (start_point + length_val)])
        for i in range(start_point, (start_point + length_val)):
            'maybe not reasonable'
            if abs(lon[i] - center_lon) <= R_limit_set and abs(lat[i] - center_lat) <= R_limit_set:
                flag = 'include in fov'
            else:
                'if not in the fov'
                flag = 'out of fov'
                start_new = i + 1
                break

        if (flag == 'include in fov'): # not Redundant
            while(flag == 'include in fov'):
                if (start_point + length_val - 1) <= (len(lon) - 2): # -2 becasue the length_val will +1
                    length_val += 1
                    center_lon = np.mean(lon[start_point: (start_point + length_val)])
                    center_lat = np.mean(lat[start_point: (start_point + length_val)])
                    for i in range(start_point, (start_point + length_val)):
                        if abs(lon[i] - center_lon) <= R_limit_set and abs(lat[i] - center_lat) <= R_limit_set:
                            flag = 'include in fov'
                        else:
                            'if not in the fov'
                            flag = 'out of fov'
                            break
                else:
                    break

            fov_zone.append([start_point, (start_point + length_val - 1)])
            start_new = start_new + length_val

    return fov_zone

def fov_center(fov_zone, lon, lat):
    lon = np.float32(lon)
    lat = np.float32(lat)
    fov_center = []
    for i in range(len(fov_zone)):
        fov_center.append([np.mean(lon[fov_zone[i][0]:(fov_zone[i][1])]), np.mean(lat[fov_zone[i][0]:(fov_zone[i][1])])])
    return fov_center

def fov_statistic(eye_lon, eye_lat , fov_center_location, fov_zone):

    'just one subject of one video'
    eye_lon = np.float32(eye_lon)
    eye_lat = np.float32(eye_lat)
    interval = 20
    n_rows = int(180 / interval )
    n_columns = int(360 / interval)
    n_cells = n_rows * n_columns
    num_fov = np.zeros((n_rows, n_columns))
    distance_fov = np.zeros((n_rows, n_columns))
    mean_fov_eye_lon = np.zeros((n_rows, n_columns))
    mean_fov_eye_lat = np.zeros((n_rows, n_columns))
    std_fov_eye_lon =  np.zeros((n_rows, n_columns))
    std_fov_eye_lat = np.zeros((n_rows, n_columns))
    num_fixation = np.zeros((n_rows, n_columns))
    std_fov_eye_distance = np.zeros((n_rows, n_columns))
    mean_fov_eye_distance = np.zeros((n_rows, n_columns))
    data_sta_lon =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]
    data_sta_lat =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]
    data_sta_distance =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]

    'count the fov number and save fov fixations'
    for i in range(n_rows):
        for j in range(n_columns):
            center_lon = ((180 - (j + 1) * interval )  + (180 - j * interval )) / 2
            center_lat = ((90 - (i + 1) * interval )  + (90 - i * interval)) / 2
            distance_fov[i][j] = cal_sphere_distance(1, center_lon, center_lat, 0, 0)
            for i_fov in range(len(fov_center_location)):
                if   (180 - (j + 1) * interval)  < fov_center_location[i_fov][0] <= (180 - j * interval)  and (90 - (i + 1) * interval)  < fov_center_location[i_fov][1]  <= (90 - i * interval):
                    num_fov[i][j] += 1
                    'save the data'
                    data_sta_lon[i][j].extend(eye_lon[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                    data_sta_lat[i][j].extend(eye_lat[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                    distance_one_fixtion = []
                    # for i in range(fov_zone[i_fov][0], fov_zone[i_fov][1]):
                    #     distance_one_fixtion.append(cal_sphere_distance(1, ))
                        # data_sta_distance[i][j].extend(eye_lat[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                        # print('i,j,i_fov: %s, %s, %s'%(i, j, i_fov))

    'statistic the mean and std'
    for i in range(n_rows):
        for j in range(n_columns):
            mean_fov_eye_lon[i][j] = np.mean(data_sta_lon[i][j])
            mean_fov_eye_lat[i][j] = np.mean(data_sta_lat[i][j])
            std_fov_eye_lon[i][j] = np.std(data_sta_lon[i][j])
            std_fov_eye_lat[i][j] = np.std(data_sta_lat[i][j])
            num_fixation[i][j] = len(data_sta_lon[i][j])

    return num_fov, mean_fov_eye_lon, mean_fov_eye_lat, std_fov_eye_lon, std_fov_eye_lat, num_fixation, distance_fov

def fov_plot_content(eye_lon, eye_lat , fov_center_location, fov_zone, save_path, video_name, subject_name):

    'just one subject of one video'
    eye_lon = np.float32(eye_lon)
    eye_lat = np.float32(eye_lat)
    interval = 20
    n_rows = int(180 / interval)
    n_columns = int(360 / interval)
    n_cells = n_rows * n_columns
    num_fov = np.zeros((n_rows, n_columns))
    distance_fov = np.zeros((n_rows, n_columns))
    mean_fov_eye_lon = np.zeros((n_rows, n_columns))
    mean_fov_eye_lat = np.zeros((n_rows, n_columns))
    std_fov_eye_lon =  np.zeros((n_rows, n_columns))
    std_fov_eye_lat = np.zeros((n_rows, n_columns))
    num_fixation = np.zeros((n_rows, n_columns))
    std_fov_eye_distance = np.zeros((n_rows, n_columns))
    mean_fov_eye_distance = np.zeros((n_rows, n_columns))
    data_sta_lon =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]
    data_sta_lat =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]
    data_sta_distance =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]

    'count the fov number and save fov fixations'
    for i in range(n_rows):
        for j in range(n_columns):
            center_lon = ((180 - (j + 1) * interval)  + (180 - j * interval)) / 2
            center_lat = ((90 - (i + 1) * interval)  + (90 - i * interval)) / 2
            distance_fov[i][j] = cal_sphere_distance(1, center_lon, center_lat, 0, 0)

            for i_fov in range(len(fov_center_location)):
                if   (180 - (j + 1) * interval)  < fov_center_location[i_fov][0] <= (180 - j * interval)  and (90 - (i + 1) * interval)  < fov_center_location[i_fov][1]  <= (90 - i * interval):
                    num_fov[i][j] += 1
                    'save the data'
                    data_sta_lon[i][j].extend(eye_lon[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                    data_sta_lat[i][j].extend(eye_lat[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                    distance_one_fixtion = []
                    # for i in range(fov_zone[i_fov][0], fov_zone[i_fov][1]):
                    #     distance_one_fixtion.append(cal_sphere_distance(1, ))
                        # data_sta_distance[i][j].extend(eye_lat[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                        # print('i,j,i_fov: %s, %s, %s'%(i, j, i_fov))

    'statistic the mean and std'
    for i in range(n_rows):
        for j in range(n_columns):
            if num_fov[i][j] != 0:
                center_lon = ((180 - (j + 1) * interval) + (180 - j *  interval)) / 2
                center_lat = ((90 - (i + 1) *  interval) + (90 - i *  interval)) / 2

                figure = plt.figure()
                ax1 = figure.add_subplot(111)
                ax1.scatter(data_sta_lon[i][j], data_sta_lat[i][j],
                            c='r', marker='.')
                plt.xlim(-55, 55)
                plt.ylim(-56.5, 56.5)
                ax1.set_xlabel('longitude/10$^\circ$')
                ax1.set_ylabel('latitude/10$^\circ$')
                ax1.set_title('fov: (%s,%s), center_location:(%s,%s),num_fixation: %s' % (str(i),str(j), str(center_lon), str(center_lat), str(len(data_sta_lon[i][j]))))

                path_figure_lat = save_path + video_name + subject_name + str(i) + str(j) + '.png'
                plt.savefig(path_figure_lat)

                # plt.show()
            mean_fov_eye_lon[i][j] = np.mean(data_sta_lon[i][j])
            mean_fov_eye_lat[i][j] = np.mean(data_sta_lat[i][j])
            std_fov_eye_lon[i][j] = np.std(data_sta_lon[i][j])
            std_fov_eye_lat[i][j] = np.std(data_sta_lat[i][j])
            num_fixation[i][j] = len(data_sta_lon[i][j])

    return num_fov, mean_fov_eye_lon, mean_fov_eye_lat, std_fov_eye_lon, std_fov_eye_lat, num_fixation, distance_fov


def fov_plot_content_all_subjects(eye_lon, eye_lat , fov_center_location, fov_zone):

    'just one subject of one video'
    eye_lon = np.float32(eye_lon)
    eye_lat = np.float32(eye_lat)

    interval = 20
    n_rows = int(180 / interval)
    n_columns = int(360 / interval)

    n_cells = n_rows * n_columns
    num_fov = np.zeros((n_rows, n_columns))
    data_sta_lon =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]
    data_sta_lat =  [[[0 for i in range(1)] for j in range(n_columns)] for l in range(n_rows)]

    'count the fov number and save fov fixations'
    for i in range(n_rows):
        for j in range(n_columns):
            for i_fov in range(len(fov_center_location)):
                if   (180 - (j + 1) * interval)  < fov_center_location[i_fov][0] <= (180 - j * interval)  and (90 - (i + 1) * interval)  < fov_center_location[i_fov][1]  <= (90 - i * interval):
                    num_fov[i][j] += 1
                    'save the data'
                    data_sta_lon[i][j].extend(eye_lon[fov_zone[i_fov][0]: fov_zone[i_fov][1]])
                    data_sta_lat[i][j].extend(eye_lat[fov_zone[i_fov][0]: fov_zone[i_fov][1]])

    return data_sta_lon, data_sta_lat, num_fov

def cal_direction_proportion_fov(lon, lat):
    'cal one fov direction ratio'
    fix_length = len(lon)
    direction_ratio = [0] * 4 # up, down, left, right

    for i in range(fix_length):
        derection = get_direction(lon[i], lat[i])

        if 315 <= derection <= 360 or 0 < derection < 45:
            direction_ratio[2] += 1
        elif 45 <= derection < 135:
            direction_ratio[0] += 1
        elif 135 <= derection < 225:
            direction_ratio[3] += 1
        elif 225 <= derection < 315:
            direction_ratio[1] += 1
        else:
            pass

    sum_fixations = np.sum(direction_ratio)
    for i in range(4):
        direction_ratio[i] = direction_ratio[i] / sum_fixations

    return direction_ratio

def get_direction(lon, lat):
    'cal the derection'
    try:
        radian = math.atan(lat / lon)
        degree = radian * 180 / np.pi
        # print(degree)
        # print(np.pi / 3)
    except ZeroDivisionError as e:
        # print(e)
        if lat > 0:
            radian = math.atan(float('Inf'))
            degree = radian * 180 / np.pi
        else:
            radian = math.atan(-float('Inf'))
            degree = radian * 180 / np.pi
    'map to the direction'
    if lon > 0 and lat > 0:
        pass
    if lon < 0 and lat > 0:
        degree = -1 * degree + 90
    if lon < 0 and lat < 0:
        degree = degree + 180
    if lon > 0 and lat < 0:
        degree = -1 * degree + 270

    return degree

def fov_tensor_in_sequce(lon, fov_zone):
    'from fov zone get a same size tensor'
    fov_tensor = np.zeros(len(lon))
    for i in range(len(fov_zone)):
        start_point = fov_zone[i][0] - 1
        end_point = fov_zone[i][1] # left closem,right open
        fov_tensor[start_point : end_point] = 1

    return fov_tensor

def cal_consist_in_sequence(fov_tensor_all):
    'fov_tensor_all should have the shape of m X n, where m is number of subjects, n is num of frames'
    subject_num = len(fov_tensor_all)
    frame_num = len(fov_tensor_all[0][:])
    public_tensor = np.zeros(frame_num)
    one = np.ones(subject_num)
    fov_tensor_all = np.array(fov_tensor_all)

    for i in range(frame_num):
        print('fov_tensor_all[:, i], one: ', end = '')
        print(fov_tensor_all[:, i])
        print(one)
        print(np.shape(fov_tensor_all[:, i]))
        if(np.array_equal(fov_tensor_all[:, i], one) == True):
            print('>>> public_tensor[i] = 1')
            public_tensor[i] = 1

    print('>>>>>>>>>>>debug here0')
    consist_tensor = []
    for num in range(subject_num):
        public_counter = 0
        for j in range(frame_num):
            if fov_tensor_all[num][j] == 1 and public_tensor[j] == 1:
                public_counter += 1
        consist = np.float32(public_counter * 1.0 / frame_num)
        consist_tensor.append(consist)

    return consist_tensor

def get_mean_and_std(one_video_data):
    'calculate the mean and std val in each frame of one video,the data should have the shape of num_subjects X num_frames'
    one_video_data = np.array(np.float32(one_video_data))
    return  np.mean(one_video_data, axis=0), np.std(one_video_data, axis=0)

def get_distance_mean_and_std(one_video_lon, one_video_lat):
    'both data have the shape of num_subjects X num_frames'
    subject_num = len(one_video_lon)
    frame_num = len(one_video_lon[0][:])

    one_video_lon = np.array(one_video_lon).astype('float32')
    one_video_lat = np.array(one_video_lat).astype('float32')
    one_video_distance = np.zeros((subject_num, frame_num))

    for j in range(frame_num):
        center_lon = np.mean(one_video_lon[:, j])
        center_lat = np.mean(one_video_lat[:, j])
        for i in range(subject_num):
            one_video_distance[i, j] = np.sqrt(np.sum(np.square(one_video_lon[i][j] - center_lon) + np.square(one_video_lon[i][j] - center_lat)))

    return  np.mean(one_video_distance, axis=0), np.std(one_video_distance, axis=0)

def get_mean_x_y(one_video_eye_x, one_video_eye_y):
    'cal the mean x,y in one video over all subjects'
    mean_x = np.mean(one_video_eye_x, axis = 1)
    mean_y = np.mean(one_video_eye_y, axis = 1)

    return np.mean(mean_x), np.mean(mean_y)

def get_eye_distanc_fov_slop(one_video_lon, one_video_lat, one_video_eye_x, one_video_eye_y):
    'cal the relationship of eye_distance with the fov change slop'
    pass

def cal_mean_std_in_fov(x_data, y_data, fov_zone):
    '''cal mean_x, mean_y, in fovzone,for one subject in one video
    note the x,y data is one subject's data'''
    mean_x = []
    std_x = []
    mean_y = []
    std_y = []

    for i in range(len(fov_zone)):
        mean_x.append(np.mean(x_data[fov_zone[i][0]:fov_zone[i][1]])) # mean of row
        std_x.append(np.std(x_data[fov_zone[i][0]:fov_zone[i][1]]))
        mean_y.append(np.mean(y_data[fov_zone[i][0]:fov_zone[i][1]])) # mean of row
        std_y.append(np.std(y_data[fov_zone[i][0]:fov_zone[i][1]]))

    return np.mean(mean_x), np.mean(std_x),np.mean(mean_y), np.mean(std_y)

def cal_mean_std_out_fov(x_data, y_data, fov_zone):
    '''cal mean_x, mean_y, in fovzone,for one subject in one video
    note the x,y data is one subject's data'''
    mean_x = []
    std_x = []
    mean_y = []
    std_y = []

    for i in range(len(fov_zone)):
        if i == 0 & fov_zone[0][0] > 1: # if only one number ,will result in nan
            # print('if_1:' + str(np.mean(x_data[0:fov_zone[0][0]])))
            mean_x.append(np.mean(x_data[0:fov_zone[0][0]]))  # mean of row
            std_x.append(np.std(x_data[0:fov_zone[0][0]]))
            mean_y.append(np.mean(y_data[0:fov_zone[0][0]]))  # mean of row
            std_y.append(np.std(y_data[0:fov_zone[0][0]]))

        elif i == (len(fov_zone) - 1) & fov_zone[i][1] < (len(y_data) - 1):
            # print('if_2:' + str(np.mean(x_data[fov_zone[i][1]:])))
            mean_x.append(np.mean(x_data[fov_zone[i][1]:]))  # mean of row
            std_x.append(np.std(x_data[fov_zone[i][1]:]))
            mean_y.append(np.mean(y_data[fov_zone[i][1]:]))  # mean of row
            std_y.append(np.std(y_data[fov_zone[i][1]:]))

        elif i < (len(fov_zone) - 1):
            if (fov_zone[i+1][0] - fov_zone[i][1]) >= 1:
                # print('if_3:' + str(np.mean(x_data[fov_zone[i][1]:fov_zone[i+1][0]])) + '_' +str(fov_zone[i+1][0]) + '_' +str(fov_zone[i][1]))
                mean_x.append(np.mean(x_data[fov_zone[i][1]:fov_zone[i+1][0]])) # mean of row
                std_x.append(np.std(x_data[fov_zone[i][1]:fov_zone[i+1][0]]))
                mean_y.append(np.mean(y_data[fov_zone[i][1]:fov_zone[i+1][0]])) # mean of row
                std_y.append(np.std(y_data[fov_zone[i][1]:fov_zone[i+1][0]]))
        else:
            pass

    return np.mean(mean_x), np.mean(std_x),np.mean(mean_y), np.mean(std_y)

def save_txt(data, save_path):
    '''save txt matrix, note that the matrix should have the shape of m X n'''
    mat = np.matrix(np.array(data))

    with open(save_path, 'wb') as f: # read and write use the same 'wb'
        for line in mat:
            np.savetxt(f, line, fmt='%.6f') # no need for f.close

def save_txt_with_blank(data, save_path):
    '''save the data with blan and \n, for excell plot'''
    data = np.array(data)

    f = open(save_path, 'w')
    for i in range(len(data)):
        for j in range(len(data[0])):
            f.write(str(data[i][j]) + '\t')
        f.write('\n')
    f.close()

def save_video_config(self,source_path, save_path, video_dic):
    game_dic_new_all = video_dic
    for i in range(len(game_dic_new_all)):
        if  i <=  len(game_dic_new_all): #len(game_dic_new_all)
            file_in_1 = source_path + str(game_dic_new_all[i]) + '.mp4'
            CONFIG_FILE = save_path + str(game_dic_new_all[i]) + '.cfg'

            # # get the paramters
            video = imageio.get_reader(file_in_1, 'ffmpeg')
            self.frame_per_second = video._meta['fps']
            self.frame_total = video._meta['nframes'] - 10
            self.frame_size = video._meta['source_size']
            self.video_size_width = int(self.frame_size[0])
            self.video_size_heigth = int(self.frame_size[1])
            self.second_total = self.frame_total / self.frame_per_second

            FRAMESCOUNT = self.frame_total
            FRAMERATE = self.frame_per_second
            IMAGEWIDTH = self.video_size_width
            IMAGEHEIGHT = self.video_size_heigth

            # write through txt
            f_config = open(CONFIG_FILE,"w")
            f_config.write("NAME\n")
            f_config.write(str(game_dic_new_all[i])+'\n')
            f_config.write("FRAMESCOUNT\n")
            f_config.write(str(FRAMESCOUNT)+'\n')
            f_config.write("FRAMERATE\n")
            f_config.write(str(FRAMERATE)+'\n')
            f_config.write("IMAGEWIDTH\n")
            f_config.write(str(IMAGEWIDTH)+'\n')
            f_config.write("IMAGEHEIGHT\n")
            f_config.write(str(IMAGEHEIGHT)+'\n')
            f_config.close()

def get_video_config(video_path, video_name):
    file_in_1 = video_path + video_name + '.mp4'
    # # get the paramters
    # # get the paramters
    video = imageio.get_reader(file_in_1, 'ffmpeg')

    FRAMERATE= round(video._meta['fps'])
    FRAMESCOUNT = video._meta['nframes']
    Frame_size  = video._meta['source_size']
    IMAGEWIDTH = round(Frame_size[0])
    IMAGEHEIGHT = round(Frame_size[1])
    Second_total = FRAMESCOUNT / FRAMERATE

    return FRAMERATE,  FRAMESCOUNT,  IMAGEWIDTH, IMAGEHEIGHT

def cal_sphere_distance(R, Lon1_degree, Lat1_degree, Lon2_degree, Lat2_degree):
    'calculate the distance in sphere between two point in (lon,lat)'
    lat1 = radians(Lat1_degree)  # change degeree to radians
    lon1 = radians(Lon1_degree)
    lon2 = radians(Lon2_degree)
    lat2 = radians(Lat2_degree)

    C = cos(lat1) * cos(lat2) * cos(lon1 - lon2) + sin(lat1) * sin(lat2)
    distance  = R * acos(C)

    return distance


def cal_2d_distance(x,y):
    'cal distance in pixel'
    return np.sqrt(x**2 + y**2)


def sta_for_gaussian(data, sta_type = 'x'):
    'statistic the number of fixation fall into specifical cells'

    data = np.array(data)
    interval_cells = 10 # pixel
    width = 1080
    height = 1200
    n_column_cells =int(width / interval_cells) # 108
    n_row_cells = int(height / interval_cells) # 120

    if sta_type == 'x':
        sta_gaussian = np.zeros(n_column_cells)

        for i in range(len(data)):
            for j in range(n_column_cells):
                if (width / 2 - (j + 1) * interval_cells) < data[i] <= (width / 2 - j * interval_cells):
                    sta_gaussian[j] += 1
                    break

    elif sta_type == 'y':
        sta_gaussian = np.zeros(n_row_cells)

        for i in range(len(data)):
            for j in range(n_row_cells):
                if (height / 2 - (j + 1) * interval_cells) < data[i] <= (height / 2 - j * interval_cells):
                    sta_gaussian[j] += 1
                    break # note this is cnecesaary to improve effcience

    return sta_gaussian

def sta_for_heatmap_0(eye_x, eye_y):
    ''' the algrithm is too slow and low efficiency'''
    'statistic the number of fixation fall into specifical cells, for 2d heatmap'
    eye_x = np.array(eye_x)
    eye_y = np.array(eye_y)
    'config'
    interval_cells = 20 # pixel
    width = 1080
    height = 1200
    n_column_cells =int(width / interval_cells) # 108
    n_row_cells = int(height / interval_cells) # 120
    sta_heatmap = np.zeros((n_row_cells, n_column_cells))

    'cal the number'
    for i in range(n_row_cells):
        for j in range(n_column_cells):
            for i_data in range(len(eye_x)):
                if (width / 2 - (j + 1) * interval_cells) < eye_x[i_data] <= (width / 2 - j * interval_cells) and \
                   (height / 2 - (i + 1) * interval_cells) < eye_y[i_data] <= (height / 2 - i * interval_cells):
                    sta_heatmap[i][j] += 1

    return sta_heatmap

def sta_for_heatmap(eye_x, eye_y):
    ''' it's more efficency '''
    'config'
    interval_cells = 10 # pixel
    width = 1080
    height = 1200
    half_point_col = int(width/(interval_cells * 2)) # 54
    half_point_row = int(height/(interval_cells * 2)) # 60

    'convert to numpy array'
    eye_x = np.array(eye_x) / interval_cells
    eye_y = np.array(eye_y) / interval_cells
    eye_x = [int(round(i)) for i in eye_x]
    eye_y = [int(round(i)) for i in eye_y]

    n_column_cells =int(width / interval_cells) # 108
    n_row_cells = int(height / interval_cells) # 120
    sta_heatmap = np.zeros((n_row_cells + 1, n_column_cells + 1))

    'cal the number'
    for i in range(len(eye_x)):
        try:
             location_col =  half_point_col - eye_x[i] # 'convert the coordinate, the leftup is (0, 0), the rightdown is (108, 120)'
             location_raw = half_point_row - eye_y[i]
             sta_heatmap[location_raw][location_col] += 1

        except Exception as e:
            print('>>>>>>>>>e: %s'%e)
            raise(e)

    return sta_heatmap

def get_y_up_ratio(data):
    'caclutale the ratio in up direction of the data of one fov, the data should be a list'
    data = np.array(data)
    num_up = []

    for i in data:
        if i < 0:
            num_up.append((i))

    ratio_up = len(num_up) / len(data)

    return ratio_up

def get_eye_velocity(eye_x, eye_y, fov_center_location, fov_zone):
    ''' cal the velocity of one subject of one video '''

    eye_x = np.array(eye_x)
    eye_y = np.array(eye_y)
    velocity_eye_x = []
    velocity_eye_y = []

    mode = 'odd frame'   # 'odd frame',  'even frame' (oushu)
    if mode == 'odd frame':
        odd_eye_x = eye_x[0::2] # eye_x[start: stop: step]
        odd_eye_y = eye_x[0::2] # eye_x[start: stop: step]

        for i in range(1, len(odd_eye_x)):
            velocity_eye_x.append((odd_eye_x[i] - odd_eye_x[i-1]))
            velocity_eye_y.append((odd_eye_y[i] - odd_eye_y[i-1]))

    if mode == 'even frame':
        odd_eye_x = eye_x[1::2] # eye_x[start: stop: step]
        odd_eye_y = eye_x[1::2] # eye_x[start: stop: step]



def fixation2salmap_fcb_2dim(fixation, mapwidth, mapheight,sigma):
    'get the fcb map, sigma in degree'
    my_sigma_in_lon = sigma
    my_sigma_in_lat = sigma
    fixation_total = np.shape(fixation)[0]
    print('>> fixation_total:%d'%fixation_total)
    salmap = np.zeros((mapwidth, mapheight))

    for x in range(mapwidth):
        for y in range(mapheight):
            cur_lon = x  - mapwidth / 2
            cur_lat = y  - mapheight / 2
            sal = math.exp(-1.0 / (2.0) * (cur_lon**2/(my_sigma_in_lon**2)+cur_lat**2/(my_sigma_in_lat**2)))
            salmap[x, y] += sal

    salmap = salmap * (1.0 / np.amax(salmap))
    salmap = np.transpose(salmap)
    return salmap


def fixation2salmap_in_pixel(fixation, mapwidth, mapheight, my_sigma = 10):
    'get the heat map, all unite in pixel'
    fixation_total = np.shape(fixation)[0] # the rows
    x_degree_per_pixel = 360.0 / mapwidth
    y_degree_per_pixel = 180.0 / mapheight
    salmap = np.zeros((mapwidth, mapheight))

    for x in range(mapwidth):
        for y in range(mapheight):
            cur_lon = x * x_degree_per_pixel - 180.0
            cur_lat = y * y_degree_per_pixel - 90.0
            for fixation_count in range(fixation_total): # for each row
                cur_fixation_lon = fixation[fixation_count][0] # the fixation lon
                cur_fixation_lat = fixation[fixation_count][1] # the fixation lat
                distance_to_cur_fixation = haversine(lon1=cur_lon,
                                                     lat1=cur_lat,
                                                     lon2=cur_fixation_lon,
                                                     lat2=cur_fixation_lat)
                distance_to_cur_fixation = distance_to_cur_fixation / math.pi * 180.0
                sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation**2) / (my_sigma**2))
                salmap[x, y] += sal
    salmap = salmap * ( 1.0 / np.amax(salmap) )
    salmap = np.transpose(salmap)
    return salmap


def fixation2salmap_sp_my_sigma(fixation, mapwidth, mapheight, my_sigma = (11.75+13.78)/2):
    'get the heat map'
    fixation_total = np.shape(fixation)[0] # the rows
    x_degree_per_pixel = 360.0 / mapwidth
    y_degree_per_pixel = 180.0 / mapheight
    salmap = np.zeros((mapwidth, mapheight))

    for x in range(mapwidth):
        for y in range(mapheight):
            cur_lon = x * x_degree_per_pixel - 180.0
            cur_lat = y * y_degree_per_pixel - 90.0
            for fixation_count in range(fixation_total): # for each row
                cur_fixation_lon = fixation[fixation_count][0] # the fixation lon
                cur_fixation_lat = fixation[fixation_count][1] # the fixation lat
                distance_to_cur_fixation = haversine(lon1=cur_lon,
                                                     lat1=cur_lat,
                                                     lon2=cur_fixation_lon,
                                                     lat2=cur_fixation_lat)
                distance_to_cur_fixation = distance_to_cur_fixation / math.pi * 180.0
                sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation**2) / (my_sigma**2))
                salmap[x, y] += sal
    salmap = salmap * ( 1.0 / np.amax(salmap) )
    salmap = np.transpose(salmap)
    return salmap

def fixation2salmap_in_pixel(fixation, mapwidth, mapheight, my_sigma):
    'get the heat map'
    fixation_total = np.shape(fixation)[0] # the rows
    salmap = np.zeros((mapwidth, mapheight))

    for x in range(mapwidth):
        for y in range(mapheight):
            cur_lon = x - mapwidth / 2
            cur_lat = y  - mapwidth / 2
            for fixation_count in range(fixation_total): # for each row
                cur_fixation_lon = fixation[fixation_count][0] # the fixation lon
                cur_fixation_lat = fixation[fixation_count][1] # the fixation lat
                distance_to_cur_fixation = cal_2d_distance_two_points(lon1=cur_lon,
                                                                         lat1=cur_lat,
                                                                         lon2=cur_fixation_lon,
                                                                         lat2=cur_fixation_lat)
                distance_to_cur_fixation = distance_to_cur_fixation / math.pi * 180.0
                sal = math.exp(-1.0 / 2.0 * (distance_to_cur_fixation**2) / (my_sigma**2))
                salmap[x, y] += sal
    salmap = salmap * ( 1.0 / np.amax(salmap) )
    salmap = np.transpose(salmap)
    return salmap


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 1.0
    return c * r

def cal_2d_distance_two_points(x_center, y_center, x_now, y_now):
    'cal distance in pixel'
    return np.sqrt((x_center - x_now)**2 + (y_center - y_now)**2)

def calc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


def get_raw_data(subject_dic, source_path, video_name):
    one_video_lon = []
    one_video_lat = []
    one_video_eye_x = []
    one_video_eye_y = []
    one_video_eye_lon = []
    one_video_eye_lat = []

    for i in range(len(subject_dic)):

        file = source_path + '//' + str(subject_dic[i]) + '//' + str(video_name) + '.txt'
        f = open(file)
        lines = f.readlines()
        raw_data = []  # notice to reset it

        for line in lines:
            line = line.split()
            raw_data.append(line)
        raw_data = np.array(raw_data)

        head_lon = raw_data[:, 2]
        head_lat = raw_data[:, 1]
        eye_x = raw_data[:, 4]
        eye_y = raw_data[:, 5]
        eye_lon = raw_data[:, 6]
        eye_lat = raw_data[:, 7]
        # save one video's data
        one_video_lon.append(head_lon)
        one_video_lat.append(head_lat)
        one_video_eye_x.append(np.round(eye_x.astype('float32')))  # scale the data
        one_video_eye_y.append(np.round(eye_y.astype('float32')))
        one_video_eye_lon.append(eye_lon.astype('float32') * 180 / np.pi)  # scale the data
        one_video_eye_lat.append(eye_lat.astype('float32') * 180 / np.pi)

    return one_video_lon, one_video_lat, one_video_eye_x, one_video_eye_y, one_video_eye_lon, one_video_eye_lat



def fov_detection(lon, lat, R_limit_set, length_val_set):
    '''detect one subject's fov'''
    # detection config
    # length_val = length_val_set  # 45 as the framerate is about 30 fps, one fov need at least 2s, the number of fixation in one fov
    # R_limit_= R_limit_set  # 3 degree

    fov_zone = []
    length_raw = length_val_set # the fixation number of a fov
    start_new = 0
    flag = 'not detect' # flag is the fixation is in fov

    lon = np.float32(lon)
    lat = np.float32(lat)

    while (start_new + length_raw - 1) < (len(lon)):
        start_point = start_new
        length_val = length_raw
        zone_delat_val = [] # save the delta value, remember to reset it

        center_lon = np.mean(lon[start_point : (start_point + length_val)])
        center_lat = np.mean(lat[start_point : (start_point + length_val)])
        for i in range(start_point, (start_point + length_val)):
            'maybe not reasonable'
            if abs(lon[i] - center_lon) <= R_limit_set and abs(lat[i] - center_lat) <= R_limit_set:
                flag = 'include in fov'
            else:
                'if not in the fov'
                flag = 'out of fov'
                start_new = i + 1
                break

        if (flag == 'include in fov'): # not Redundant
            while(flag == 'include in fov'):
                if (start_point + length_val - 1) <= (len(lon) - 2): # -2 becasue the length_val will +1
                    length_val += 1
                    center_lon = np.mean(lon[start_point: (start_point + length_val)])
                    center_lat = np.mean(lat[start_point: (start_point + length_val)])
                    for i in range(start_point, (start_point + length_val)):
                        if abs(lon[i] - center_lon) <= R_limit_set and abs(lat[i] - center_lat) <= R_limit_set:
                            flag = 'include in fov'
                        else:
                            'if not in the fov'
                            flag = 'out of fov'
                            break
                else:
                    break

            fov_zone.append([start_point, (start_point + length_val - 1)])
            start_new = start_new + length_val

    return fov_zone

def fov_center(fov_zone, lon, lat):
    lon = np.float32(lon)
    lat = np.float32(lat)
    fov_center = []
    for i in range(len(fov_zone)):
        fov_center.append([np.mean(lon[fov_zone[i][0]:(fov_zone[i][1])]), np.mean(lat[fov_zone[i][0]:(fov_zone[i][1])])])
    return fov_center
