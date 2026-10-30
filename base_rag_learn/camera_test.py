
import cv2  # 导入OpenCV库，用于摄像头采集、视频显示和窗口状态检查


# 1. 打开已经测试成功的DroidCam摄像头
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

# 2. 设置窗口名称
window_name = "DroidCam - OpenCV"

if not cap.isOpened():
    print("摄像头打开失败！")

else:
    print("摄像头打开成功！")

    # 创建OpenCV窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    first_frame = True

    try:
        while True:


            # 读取一帧视频
            ret, frame = cap.read()
            if first_frame:
                print("图像形状：", frame.shape)
                print("图像高度：", frame.shape[0])
                print("图像宽度：", frame.shape[1])
                print("通道数量：", frame.shape[2])
                first_frame = False
            if not ret:
                print("摄像头画面读取失败！")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 显示当前视频帧
            cv2.imshow(window_name, frame)

            # 方法一：按Q键退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("用户按下Q键，退出程序")
                break

            # 方法二：点击窗口右上角×退出
            try:
                # WND_PROP_VISIBLE表示窗口是否仍然可见
                # 小于1说明窗口已经被关闭
                if cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE
                ) < 1:
                    print("用户关闭窗口，退出程序")
                    break

            except cv2.error:
                # 如果窗口已经销毁，查询窗口属性可能报错
                # 此时直接结束循环
                break

    finally:
        # 无论通过哪种方式退出，都释放摄像头
        cap.release()

        # 关闭OpenCV创建的所有窗口
        cv2.destroyAllWindows()
