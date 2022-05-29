import cv2
import torch
import time
from model import MattingNetwork
from PIL import Image
from torchvision import transforms
# from threading import Thread, Lock


# ----------- Utility classes -------------


class Camera:
	"""
	A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
	Use .read() in a tight loop to get the newest frame.
	"""

	def __init__(self, device_id=0, width=1280, height=720):
		self.capture = cv2.VideoCapture(device_id)
		# self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		# self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		self.width = int(width)
		self.height = int(height)
		# self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
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
		# with self.read_lock:
		grabbed, frame = self.capture.read()
		# self.frame = frame
		# frame = self.frame
		return frame

	def __exit__(self, exec_type, exc_value, traceback):
		self.capture.release()


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


class Displayer:
	"""
	Wrapper for playing a stream with cv2.imshow().
	It also tracks FPS and optionally overlays info onto the stream.
	"""

	def __init__(self, title, width=None, height=None, show_info=True):
		self.title, self.width, self.height = title, width, height
		# self.show_info = show_info
		# self.fps_tracker = FPSTracker()
		# cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
		# if width is not None and height is not None:
		# 	cv2.resizeWindow(self.title, width, height)

	# Update the currently showing frame and return key press char code
	def step(self, image):
		# fps_estimate = self.fps_tracker.tick()
		# if self.show_info and fps_estimate is not None:
		# 	message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
		# 	cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
		cv2.imshow(self.title, image)
		return cv2.waitKey(1) & 0xFF


def cv2_frame_to_cuda(frame):
	"""
	convert cv2 frame to tensor.
	"""
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	loader = transforms.ToTensor()
	return loader(Image.fromarray(frame)).to(device, dtype, non_blocking=True).unsqueeze(0)


def auto_downsample_ratio(h, w):
	"""
	Automatically find a downsample ratio so that the largest side of the resolution be 512px.
	"""
	return min(512 / max(h, w), 1)


# --------------- Main ---------------

if __name__ == '__main__':

	width, height = (1920, 1080)  # the show windows size.
	output_background = 'green'  # Options: ["green", "white", "image"].
	dtype = torch.float32
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load mobilenetv3 model
	model = MattingNetwork('mobilenetv3')
	model = model.to(device, dtype, non_blocking=True).eval()
	model.load_state_dict(torch.load('./rvm_mobilenetv3.pth'))
	# model = torch.jit.script(model)
	# model = torch.jit.freeze(model)

	# cam = Camera('http://admin:admin@shj.local:8081', width=width, height=height)
	cam = cv2.VideoCapture('http://admin:admin@shj.local:8081')
	# dsp = Displayer('VideoMatting', cam.width, cam.height, show_info=True)
	print("init1")
	bgr = None
	if output_background == 'white':
		bgr = torch.tensor([255, 255, 255], device=device, dtype=dtype).div(255).view(3, 1, 1)  # white background
	elif output_background == 'green':
		bgr = torch.tensor([0, 255, 0], device=device, dtype=dtype).div(255).view(3, 1, 1)  # green background
	with torch.no_grad():
		while True:
			# matting
			print("000")
			_, frame = cam.read()
			print("999")
			src = cv2_frame_to_cuda(frame)
			rec = [None] * 4
			# downsample_ratio = auto_downsample_ratio(*src.shape[2:])
			downsample_ratio = 0.25
			fgr, pha, *rec = model(src, *rec, downsample_ratio)

			# if bgr is None:
			# 	h, w = src.shape[2:]
			# 	# print(h, w)
			# 	transform = transforms.Compose([
			# 		transforms.Resize(size=(h, w)),
			# 		transforms.ToTensor()
			# 	])
				# img = Image.open("work/background/background3.jpg")
				# bgr = transform(img).to(device, dtype, non_blocking=True)
			print('tuns')
			# com = fgr * pha + bgr * (1 - pha)
			com = 255*fgr.cpu().permute(0, 2, 3, 1).numpy()[0]
			comm = cv2.cvtColor(com, cv2.COLOR_RGB2BGR).copy()
			# key = dsp.step(com)
			print(comm.shape, comm.max(), comm.min())
			# cv2.imshow('ooo',comm)
			print("111")
			cv2.waitKey(1)
			# if key == ord('b'):
			# 	break
			# elif key == ord('q'):
			# 	exit()