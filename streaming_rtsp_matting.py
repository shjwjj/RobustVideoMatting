#!/usr/bin/env python
import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
 
 
def main():
    
    # jetson 平台
    out_send = cv2.VideoWriter('appsrc is-live=true ! videoconvert ! \
                                omxh264enc bitrate=12000000 ! video/x-h264, \
                                stream-format=byte-stream ! rtph264pay pt=96 ! \
                                udpsink host=127.0.0.1 port=5400 async=false',
                                cv2.CAP_GSTREAMER, 0, 30, (1920 ,1080), True)
    # dGPU 平台
    # out_send = cv2.VideoWriter('appsrc is-live=true  ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! \
    #     nvv4l2h264enc ! h264parse ! rtph264pay pt=96 ! udpsink host=127.0.0.1 port=5400', \
    #         cv2.CAP_GSTREAMER, 0, 25, (1280,720), True)
 
    if not out_send.isOpened():
        print('VideoWriter not opened')
        exit(0)
 
    rtsp_port_num = 8554 
 
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch("(udpsrc name=pay0 port=5400 buffer-size=524288 \
                        caps=\"application/x-rtp, media=video, clock-rate=90000, \
                        encoding-name=(string)H264, payload=96 \")")
                        
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
 
    # 输出rtsp码流信息
    print("\n *** Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)    
 
    cap = cv2.VideoCapture('http://admin:admin@shj.local:8081')
    # cap = cv2.VideoCapture('')
    import torch
    import numpy as np
    from model import MattingNetwork
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from inference_utils import VideoReader

    model = MattingNetwork('mobilenetv3').eval().cuda()  # 或 "resnet50"
    model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # 绿背景
    rec = [None] * 4                                       # 初始循环记忆（Recurrent States）
    downsample_ratio = 0.25   
    # reader = VideoReader('http://admin:admin@shj.local:8081', transform=ToTensor())

    while True:
    # for src in DataLoader(reader):
        _, mat = cap.read()
        # mat = np.transpose(mat, (2,0,1))
        # mat = np.expand_dims(mat, axis=0)
        print('Flag:   ', mat.shape)
        # fgr, pha, *rec = model(torch.from_numpy(mat).cuda(), *rec, downsample_ratio)  # 将上一帧的记忆给下一帧
        # com = fgr * pha + bgr * (1 - pha)
        # print(com.shape)
        cv2.imshow("mat", mat)

        # out_send.write(mat)
        cv2.waitKey(30) 
        
if __name__ == '__main__':
    main()


    # cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -D CUDA_ARCH_BIN='7.2' -D WITH_CUDA=1 -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D OPENCV_GENERATE_PKGCONFIG=1 -D WITH_GTK_2_X=ON -D WITH_GSTREAMER=ON..
