#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  20 02:07:13 2019
@author: prabhakar
"""
# import necessary argumnets 
import gi
import cv2
import argparse
import torch
import time
import numpy as np
from model import MattingNetwork
from torchvision.transforms import ToTensor
from threading import Thread, Lock
# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

model = MattingNetwork('mobilenetv3').eval().cuda()  # 或 "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

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
			with self.read_lock:
				self.success_reading = grabbed
				self.frame = frame

	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
		return frame

	def __exit__(self, exec_type, exc_value, traceback):
		self.capture.release()

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        # self.cap = cv2.VideoCapture(opt.device_id)
        self.cap = Camera('http://admin:admin@shj.local:8081')
        self.fpstracker = FPSTracker()
        # self.cap = Camera()
        print(self.cap.width, self.cap.height)
        self.fps_tracker = FPSTracker()
        self.number_frames = 0
        self.fps = 15
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.rec = [None] * 4                                       # 初始记忆
        self.bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # 绿背景
        # self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
        #                      'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
        #                      '! videoconvert ! video/x-raw,format=I420 ' \
        #                      '! x264enc speed-preset=ultrafast tune=zerolatency ' \
        #                      '! rtph264pay config-interval=1 name=pay0 pt=96' \
        #                      .format(opt.image_width, opt.image_height, self.fps)
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert  ' \
                             '! omxh264enc bitrate=12000000 ! video/x-h264, stream-format=byte-stream' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(self.cap.width, self.cap.height, self.fps)
    # method to capture the video feed from the camera and push it to the
    # streaming buffer.
    def on_need_data(self, src, length):
        # if self.cap.isOpened():
        if 1:
            # ret, frame = self.cap.read()
            frame = self.cap.read()
            if 1:
                with torch.no_grad():
                    # It is better to change the resolution of the camera 
                    # instead of changing the image shape as it affects the image quality.
                    st = time.time()
                    mat = cv2.resize(frame, (opt.image_width, opt.image_height), \
                        interpolation = cv2.INTER_LINEAR)
                    mat = ToTensor()(frame).unsqueeze(0)
                    # print('--mat: ', mat.shape)
                    fgr, pha, *self.rec = model(mat.cuda(), *self.rec, downsample_ratio=0.25)  # 将上一帧的记忆给下一帧
                    # print('--run')
                    # print(fgr.shape)
                    com = fgr * pha + self.bgr * (1 - pha)
                    com = com.cpu().numpy()[0].transpose((1,2,0))*255.
                    com = com.astype(np.uint8)
                    frame = com
                    print("dut:  ", time.time()-st)
            fps_estimate = self.fps_tracker.tick()
            print(frame.shape, frame.dtype, fps_estimate)
            data = frame.tostring()
            # data = frame.tostring()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                self.duration,
                                                                                self.duration / Gst.SECOND))
            if retval != Gst.FlowReturn.OK:
                print(retval)
    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(opt.port))
        self.get_mount_points().add_factory(opt.stream_uri, self.factory)
        self.attach(None)

# getting the required information from the user 
parser = argparse.ArgumentParser()
parser.add_argument("--device_id", required=False,default=0, help="device id for the \
                video device or video file location")
parser.add_argument("--fps", required=False, default=10, help="fps of the camera", type = int)
parser.add_argument("--image_width", required=False,default=640, help="video frame width", type = int)
parser.add_argument("--image_height", default=480, help="video frame height", type = int)
parser.add_argument("--port", default=8554, help="port to stream video", type = int)
parser.add_argument("--stream_uri", default = "/video_stream", help="rtsp video stream uri")
opt = parser.parse_args()

try:
    opt.device_id = int(opt.device_id)
except ValueError:
    pass

# initializing the threads and running the stream on loop.
GObject.threads_init()
Gst.init(None)
server = GstServer()
loop = GObject.MainLoop()
loop.run()
