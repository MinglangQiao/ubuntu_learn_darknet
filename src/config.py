# encoding=utf-8
from video_subject import f_game_dic_new_all, subjects_dic_all, f_game_dic_new_test, detect_threshold


fov_width = 1080 # pixel
fov_heigth = 1200
fov_step = 20
saliency_width = int(fov_width / fov_step)
saliency_height = int(fov_heigth / fov_step)

video_dic = f_game_dic_new_all
subject_dic = subjects_dic_all
video_test = f_game_dic_new_test


source_path = 'I:\\2_0925_全景视频的眼动研究\\VR_杨燕丹\\filtered_Data'

path_x = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\数据\\one_video_52_subjects_eye_x\\'
path_y = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\数据\\one_video_52_subjects_eye_y\\'
path_lon = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\数据\\one_video_52_subjects_head_lon\\'
path_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\数据\\one_video_52_subjects_head_lat\\'
path_3D = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\数据\\one_video_52_subjects_3D\\'
path_fov_one_subject = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第9周安排\\data\\fov_one_subject\\'
path_fov_one_video = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\\数据\\截取fov_单个video\\'
path_distance_mean_std = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV(11_1~12_31)\\进度监督\\第8周安排\\数据\\距离的均值和方差\\'
path_fov_head_eye = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\head_eye\\'
path_fov_head_add_eye = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\head_add_eye\\'
path_fov_head_add_eye_lon = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\path_fov_head_add_eye_lon\\'
path_fov_head_add_eye_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\path_fov_head_add_eye_lat\\'
path_fov_head_add_eye_lon_lat_subplot = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\数据和图\\path_fov_head_add_eye_lon_lat_subplot\\'
path_fov_head_add_eye_lon_lat_not_add = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\path_fov_head_add_eye_lon_lat_not_add\\'

'week_10'
fov_mean_std_x_y = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\第10周安排\\数据\\fov内的mean_std_x_y\\'


'data for HA_HB CC'
one_video_head_lon = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\txt_data\\one_video_head_lon\\'
one_video_head_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\txt_data\\one_video_head_lat\\'
one_video_eye_x = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\txt_data\\one_video_eye_x\\'
one_video_eye_y = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\txt_data\\one_video_eye_y\\'
one_video_eye_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\txt_data\\one_video_eye_lat\\'
one_video_eye_lon = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\txt_data\\one_video_eye_lon\\'


'convert frames'
video_path = '/media/ml/Data1/2_0925_全景视频的眼动研究/VR_杨燕丹/Video_All/'
frame_path = 'D:\\VR\\frames\\'

'fov统计'
fov_detection1 = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\fov截取图\\'
path_std_fov_eye_lon = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\std_fov_eye_lon\\'
path_std_fov_eye_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\std_fov_eye_lat\\'
path_std_fov_eye_lon_one_video = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\std_fov_eye_lon_one_video\\'
path_std_fov_eye_lat_one_video = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\std_fov_eye_lat_one_video\\'
path_fov_cell_subplot = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\fov划分统计\\'
path_fov_distance_mean_lon_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\fov_distance_and_mean_lon_lat_subplot\\'
path_cc_mean_lon_distance = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\path_cc_mean_lon_distance\\'
path_cc_mean_lat_distance = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\path_cc_mean_lat_distance\\'

path_fov_distance_std_lon_lat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\fov_distance_and_std_lon_lat_subplot\\'
path_for_fov_plot = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\每个fov内的fixationplot\\'
fov_plot_all_subject = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\每个fov内的fiation_所有subject\\'


path_direction_ratio_distance = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\direction_ratio_distance\\'

'all video'
# path_all_video_eye_x = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\针对所有视频\\经度数据\\'
# path_all_video_eye_y = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\针对所有视频\\纬度数据\\'
# path_all_video_fov_num = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\针对所有视频\\FOV数目的数据\\'
# path_all_video_plot = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\针对所有视频\\画图\\'


templat = 'I:\\2_0925_全景视频的眼动研究\\Salience_of_FOV\\进度监督\\数据和图\\fov截取统计\\第12周\\'

'week 12'
save_new_fov_detection = templat + 'fov_detection图\\'

path_all_video_eye_x = templat + 'x数据\\'
path_all_video_eye_y = templat + 'y数据\\'
path_all_video_fov_num = templat + 'fov数目的数据\\'
path_all_video_plot = templat + '画图\\'

'the subject dic of group_a and group_b'

subject_group_a = [0, 1, 2, 4, 8, 10, 11, 14, 17, 18, 20, 21, 23, 25, 26, 27, 28, 32, 34, 36, 37, 39, 41, 44, 49, 51] # [2, 19, 16, 4, 36, 49, 47, 38, 32, 3, 17, 39, 34, 22, 18, 43, 48, 40, 28, 14, 29, 23, 33, 12, 27, 25]
subject_group_b = [3, 5, 6, 7, 9, 12, 13, 15, 16, 19, 22, 24, 29, 30, 31, 33, 35, 38, 40, 42, 43, 45, 46, 47, 48, 50] # [0, 1, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 21, 24, 26, 30, 31, 35, 37, 41, 42, 44, 45, 46, 50, 51]


#  >>>>>>>>>>> for ubuntu
