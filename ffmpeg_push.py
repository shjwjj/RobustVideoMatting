import cv2
import subprocess as sp
import torch
import time
import numpy as np
from model import MattingNetwork
from torchvision.transforms import ToTensor
from threading import Thread, Lock

model = MattingNetwork('resnet50').eval().cuda()  # 或 "resnet50"
model.load_state_dict(torch.load('rvm_resnet50.pth'))

class FPSTracker:
    """
    An FPS tracker that computes exponentialy moving average FPS.
    """

    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (
                    1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


class Camera:
    """
    A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
    Use .read() in a tight loop to get the newest frame.
    """

    def __init__(self, device_id=0, width=640, height=480):
        self.capture = cv2.VideoCapture(device_id)
        # self.cap = Camera('http://admin:admin@shj.local:8081')
        # self.cap = Camera(0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.width = int(width)
        # self.height = int(height)
        # self.capture.set(cv.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            frame = self.capture.read()
            with torch.no_grad():
                # It is better to change the resolution of the camera 
                # instead of changing the image shape as it affects the image quality.
                # mat = cv2.resize(frame, (opt.image_width, opt.image_height), \
                #     interpolation = cv2.INTER_LINEAR)
                mat = ToTensor()(frame).unsqueeze(0)
                # print('--mat: ', mat.shape)
                fgr, pha, *rec = model(mat.cuda(), *self.rec, downsample_ratio=0.25)  # 将上一帧的记忆给下一帧
                # print('--run')
                # print(fgr.shape)
                com = fgr * pha + self.bgr * (1 - pha)
                com = com.cpu().numpy()[0].transpose((1,2,0))*255.
                com = com.astype(np.uint8)
                frame = com
                fps_estimate = self.fps_tracker.tick()
                print(frame.shape, frame.dtype, fps_estimate)
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()

rtspUrl = 'rtsp://192.168.0.128:8554/test' #这里改成本地ip，端口号不变，文件夹自定义

# 视频来源 地址需要替换自己的可识别文件地址
# filePath='D:WorkBeltDefectDetection'
# camera = cv2.VideoCapture(filePath+'\'+'Video.avi') # 从文件读取视频
# camera = cv2.VideoCapture(0) # 从文件读取视频
camera = Camera(0)
# 视频属性
size = (int(camera.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])
fps = camera.capture.get(cv2.CAP_PROP_FPS)  # 30p/self
fps = int(fps)
hz = int(1000.0 / fps)
print('size:'+ sizeStr + ' fps:' + str(fps) + ' hz:' + str(hz))

# 视频文件输出
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(filePath+'res_mv.avi',fourcc, fps, size)
# 直播管道输出
# ffmpeg推送rtmp 重点 ： 通过管道 共享数据的方式
command = [
    'ffmpeg',
    # 're',#
    # '-y', # 无需询问即可覆盖输出文件
    '-f', 'rawvideo', # 强制输入或输出文件格式
    # '-vcodec','rawvideo', # 设置视频编解码器。这是-codec:v的别名
    '-pix_fmt', 'bgr24', # 设置像素格式
    '-s', sizeStr, # 设置图像大小
    '-r', str(fps), # 设置帧率
    '-i', '-', # 输入
    # '-c:v', 'libx264',
    '-c:v', 'h264_v4l2m2m',
    '-pix_fmt', 'yuv420p',
    # '-preset', 'ultrafast',
    '-f', 'rtsp',# 强制输入或输出文件格式
    rtspUrl]

#管道特性配置
# pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
pipe = sp.Popen(command, stdin=sp.PIPE) #,shell=False
# pipe.stdin.write(frame.tostring())
while (camera.capture.isOpened()):
    frame = camera.read() # 逐帧采集视频流
    # if not ret:
    #     break
    ############################图片输出
    # 结果帧处理 存入文件 / 推流 / ffmpeg 再处理
    pipe.stdin.write(frame.tostring())  # 存入管道用于直播
    # out.write(frame)    #同时 存入视频文件 记录直播帧数据

camera.release()
out.release()