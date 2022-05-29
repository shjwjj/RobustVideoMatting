import cv2 as cv
import gi
import time
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
import torch
import numpy as np
from model import MattingNetwork
from torchvision.transforms import ToTensor
from threading import Thread, Lock
import copy



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
		self.capture = cv.VideoCapture(device_id)
		self.capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
		self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
		self.width = int(width)
		self.height = int(height)
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

def main():
	# reader = VideoReader('/home/nvidia/Documents/projects/ue4_matting/RobustVideoMatting-master/testmaskvideo.mp4', transform=ToTensor())
	# model = MattingNetwork('mobilenetv3').eval().cuda()  # 或 "resnet50"
	# model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
		# use_rtspserver = True

	# if use_rtspserver:
	# out_send = cv.VideoWriter('appsrc is-live=true ! videoconvert ! \
	# 								omxh264enc bitrate=12000000 ! video/x-h264, \
	# 								stream-format=byte-stream ! rtph264pay pt=96 ! \
	# 								udpsink host=127.0.0.1 port=5400 async=false',
	# 								cv.CAP_GSTREAMER, 0, 15, (960 ,540), True)
	# # dGPU 平台
	# # out_send = cv2.VideoWriter('appsrc is-live=true  ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! \
	# #     nvv4l2h264enc ! h264parse ! rtph264pay pt=96 ! udpsink host=127.0.0.1 port=5400', \
	# #         cv2.CAP_GSTREAMER, 0, 25, (1280,720), True)

	# if not out_send.isOpened():
	# 	print('VideoWriter not opened')
	# 	exit(0)

	# rtsp_port_num = 8554 

	# server = GstRtspServer.RTSPServer.new()
	# server.props.service = "%d" % rtsp_port_num
	# server.attach(None)

	# factory = GstRtspServer.RTSPMediaFactory.new()
	# factory.set_launch("(udpsrc name=pay0 port=5400 buffer-size=524288 \
	# 					caps=\"application/x-rtp, media=video, clock-rate=90000, \
	# 					encoding-name=(string)H264, payload=96 \")")
						
	# factory.set_shared(True)
	# server.get_mount_points().add_factory("/ds-test", factory)

	# # 输出rtsp码流信息
	# print("\n *** Launched RTSP Streaming at rtsp://127.0.0.1:%d/ds-test ***\n\n" % rtsp_port_num)    
		


	# bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # 绿背景
	rec = [None] * 4                                       # 初始记忆
	# video = cv2.VideoCapture('http://admin:admin@shj.local:8081')
	# video = cv2.VideoCapture(0)
	video = Camera('http://admin:admin@shj.local:8081')
	# for src in DataLoader(reader):
	fps_tracker = FPSTracker()
	# with torch.no_grad():
	while True: 
		# _, mat = video.read()
		mat = video.read()
		# mat = ToTensor()(mat).unsqueeze(0)
		# # print('--mat: ', mat.shape)
		# fgr, pha, *rec = model(mat.cuda(), *rec, downsample_ratio=0.25)  # 将上一帧的记忆给下一帧
		# # print('--run')
		# # print(fgr.shape)
		# com = fgr * pha + bgr * (1 - pha)
		# com = com.cpu().numpy()[0].transpose((1,2,0))*255.
		# com = com.astype(np.uint8)
		com = mat
		fps_estimate = fps_tracker.tick()
		print(com.shape, com.dtype, fps_estimate) 
		cv.imshow('mat', np.zeros([640,960,3]))
		# cv2.imwrite('./test.jpg',com)

		# if use_rtspserver:
		# out_send.write(com.copy())
		cv.waitKey(1)

if __name__ == '__main__':
    main()
