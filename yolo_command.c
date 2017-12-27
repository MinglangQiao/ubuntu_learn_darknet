网址：
https://github.com/philipperemy/yolo-9000
https://pjreddie.com/darknet/yolo/

1: get the video frame(80 class), 0.7 need to chang to another

# get frame
./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights  -prefix output /media/ml/Data1/2_0925_全景视频的眼动研究/VR_杨燕丹/Video_All/VRBasketball.mp4 -thresh 0.6

# get video
ffmpeg -framerate 25 -i output_%08d.jpg output.mp4

#remove the frame
rm output_*.jpg

2: get one frame

./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg -thresh 0.25

//./darknet detector demo  cfg/yolo.cfg yolo.weights data/dog.jpg -thresh 0.25

3:打开摄像头
在ubuntu软件里搜索cheese
或者命令行： cheese

4：摄像头识别

./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights


5:运行C语言程序

gcc get_box.c -o get_box
./get_box

# 6: 查找函数调用
# 我在前没有在整个人darknet里面查找， 只是在src里面找也是不对的
grep -nrH 'save_image'
