# 读取并显示图像
def showImgUseCV2(name):
    import cv2
    img = cv2.imread(name,0)
# img = cv2.imread('image.png',cv2.IMREAD_COLOR) # 彩色模式忽略透明度
# img = cv2.imread('image.png',cv2.IMREAD_GRAYSCALE) # 灰度模式
# img = cv2.imread('image.png',cv2.IMREAD_UNCHANGED) # 彩色图像包含alpha透明度
    cv2.imshow('image',img) # 显示图像（‘窗口名’，图像）
    cv2.waitKey(0) # 键盘绑定函数
    cv2.destroyAllWindows() # 删除所有窗口（窗口名）

# 使用matplotlib显示图像
def showImgUseMatplotlib(name):
    import cv2
    img = cv2.imread(name,0)
    from matplotlib import pyplot as plt
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# 从摄像头读取视频
def readVideoUseCV2FromCamera():
    import cv2
    # 创建VideoCapture对象，设备索引号为0.
    cap = cv2.VideoCapture(0)
    while(True):
        # 一帧一帧获取图像
        ret,frame = cap.read()
        # 显示图像
        cv2.imshow('frame',frame)
        # cap.get() 获得视频参数
        # cap.set() 设置视频参数
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 点击退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 从视频中读取视频
def readVideoUseCV2FromVideo(name):
    import cv2
    cap = cv2.VideoCapture(name)
    while cap.isOpened():
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 写入视频
def writeVideoUseCV2(name):
    import cv2
    cap = cv2.VideoCapture(0)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(name,fourcc,20.0,(640,480))

    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret==True:
            # 旋转每一帧
            # frame = cv2.flip(frame,0)
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 简单绘图
def drawSampleGraphUseCV2():
    import cv2
    import numpy as np
    img = np.zeros((512,512,3),np.uint8)

    # img：你想要绘制图形的那幅图像。
    # color：形状的颜色。以RGB
    # 为例，需要传入一个元组，例如：（255, 0, 0）
    # 代表蓝色。对于灰度图只需要传入灰度值。
    # thickness：线条的粗细。如果给一个闭合图形设置为 - 1，那么这个图形
    # 就会被填充。默认值是1.
    # linetype：线条的类型，8
    # 连接，抗锯齿等。默认情况是8
    # 连接。cv2.LINE_AA
    # 为抗锯齿，这样看起来会非常平滑。

    cv2.line(img,(0,0),(511,511),(255,0,0),5)
    cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'HMY', (10, 500), font, 4, (255, 255, 255), 2)

    winname = 'example'
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

# 使用鼠标简单绘图
def drawGraphUseMouse():
    import numpy as np
    import cv2
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(img, (x, y), 20, (255, 0, 0), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('opencv')
    cv2.setMouseCallback('opencv', draw_circle)
    while (1):
        cv2.imshow('opencv', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# 使用鼠标简单绘图
def drawGraphUseMouse2():
    import numpy as np
    import cv2

    drawing = False
    mode = True
    ix, iy = -1, -1

    def draw_circle(event, x, y, flags, param):
        global drawing, mode, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 2)
                else:
                    cv2.circle(img, (x, y), 50, (0, 255, 0), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('opencv')
    cv2.setMouseCallback('opencv', draw_circle)
    while (1):
        cv2.imshow('opencv', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('m'):
            mode = not mode
    cv2.imwrite('/home/wl/1.jpg', img)
    cv2.destroyAllWindows()

# 控制滑块更改颜色
def drawColorUseCV2():
    # coding: utf-8
    # !/usr/bin/python
    import os
    import sys
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    def nothing(x):
        pass

    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    switch = '0:OFF\n1:ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)
    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(1)
        if k == 27:
            break

        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]
    cv2.destroyAllWindows()