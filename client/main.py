import time
import socket, io, struct
import cv2, numpy as np
from device import SG90
import FaceDetector
from log import mylog
 
# 准备相机
camera=cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()
# 建立连接
client_socket = socket.socket()
client_socket.connect(('114.55.36.8', 11581))
print("linked")
# 主函数
try:
    while True:
        # 人脸检测
        ref, picture = camera.read()
        print("capture")
        if ref and inference(picture):
            print("face exist")
            # 流式传输
            stream = io.BytesIO()
            stream.write(cv2.imencode(".jpg",picture)[1].tobytes()) # 编码
            print("images encoded")
            client_socket.send(struct.pack('I', stream.tell())) # 发送流长度
            print(f"len sent = {stream.tell()}")
            time.sleep(1) # 缓冲
            stream.seek(0) 
            client_socket.sendall(stream.read()) # 发送主体数据
            print("data sent")
            stream.seek(0) 
            stream.truncate() # 重置流
            # 接收相似度
            similar = struct.unpack('d', client_socket.recv(1024))[0]
            print(f"get similar = {similar}")
            if similar > 0.7:
                # 重置相机
                camera.release()
                camera=cv2.VideoCapture(0)
                print("camera reseted")
                # 舵机转向
                SG90()
                print("door opened")
                #写入本地日志
                mylog(similar)
                print("logged")
# 退出
except KeyboardInterrupt:
    print("exist")
    camera.release()
    client_socket.send(struct.pack('I', 0))
    client_socket.close()