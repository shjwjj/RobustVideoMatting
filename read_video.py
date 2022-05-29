import cv2

# cam = cv2.VideoCapture('rtsp://127.0.0.1:8554/ds-test ')
cam = cv2.VideoCapture('rtsp://127.0.0.1:8554/video_stream ')
while 1:
    _, mat = cam.read()
    print(mat.shape)
    cv2.imshow('111',mat)
    cv2.waitKey(1)