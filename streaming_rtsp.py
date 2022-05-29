#!/usr/bin/env python
import cv2
# import gi
# gi.require_version('Gst', '1.0')
# gi.require_version('GstRtspServer', '1.0')
# from gi.repository import GObject, Gst, GstRtspServer
 
 
def main():
    
    # jetson 平台
    out_send = cv2.VideoWriter('appsrc is-live=true ! videoconvert ! \
                                omxh264enc bitrate=12000000 ! video/x-h264, \
                                stream-format=byte-stream ! rtph264pay pt=96 ! \
                                udpsink host=192.168.0.128 port=1935',
                                cv2.CAP_GSTREAMER, 0, 15, (640 ,480), True)
    # dGPU 平台
    # out_send = cv2.VideoWriter('appsrc is-live=true  ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! \
    #     nvv4l2h264enc ! h264parse ! rtph264pay pt=96 ! udpsink host=127.0.0.1 port=5400', \
    #         cv2.CAP_GSTREAMER, 0, 25, (1280,720), True)

    if not out_send.isOpened():
        print('VideoWriter not opened')
        exit(0)

    # rtsp_port_num = 8554 

    # server = GstRtspServer.RTSPServer.new()
    # server.props.service = "%d" % rtsp_port_num
    # server.attach(None)
    
    # factory = GstRtspServer.RTSPMediaFactory.new()
    # factory.set_launch("(udpsrc name=pay0 port=5400 buffer-size=524288 \
    #                     caps=\"application/x-rtp, media=video, clock-rate=90000, \
    #                     encoding-name=(string)H264, payload=96 \")")
                        
    # factory.set_shared(True)
    # server.get_mount_points().add_factory("/ds-test", factory)

    # 输出rtsp码流信息
    # print("\n *** Launched RTSP Streaming at rtsp://127.0.0.1:%d/ds-test ***\n\n" % rtsp_port_num)    
 
    # cap = cv2.VideoCapture('http://admin:admin@shj.local:8081')
    cap = cv2.VideoCapture(0)

    while True:
        _, mat = cap.read()
        cv2.imshow("mat", mat)
        print('Flag:   ', mat.shape,mat.dtype)
        out_send.write(mat)
        cv2.waitKey(30) 
        
if __name__ == '__main__':
    main()


    # cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -D CUDA_ARCH_BIN='7.2' -D WITH_CUDA=1 -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D OPENCV_GENERATE_PKGCONFIG=1 -D WITH_GTK_2_X=ON -D WITH_GSTREAMER=ON..
