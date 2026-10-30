import cv2  # 导入OpenCV库，用于摄像头采集、视频显示和窗口状态检查
import mediapipe as mp # 导入MediaPipe库，用于加载手部关键点模型，并检测手部位置
import time # 导入时间模块，为视频检测生成单调递增的时间戳
import math # 导入数学模块，用于计算关节角度
from pathlib import Path # 导入路径工具，用于定位手部检测模型
def calculate_angle(a,b,c):
    # a,b,c分别代表三个点的像素坐标
    # b是我们需要计算夹角的顶点
    # 1.计算从B指向A的方向角
    angle1 = math.atan2(a[1] - b[1], a[0] - b[0])
    # 2.计算从B指向C的方向角
    angle2 = math.atan2(c[1] - b[1], c[0] - b[0])

    # 3.两个方向角相减，并转换为角度
    angle = abs(math.degrees(angle1 - angle2))
    # 4.保证最终夹角在0-180度之间
    if angle >180:
        angle = 360 - angle
    # 5.最终返回结果
    return angle

# 调用时才打开摄像头，并运行已有的手势检测流程
def run_hand_detection(on_wake=None, is_busy=None):
    # 1.定位预先下载的手部检测模型
    model_path = Path(__file__).resolve().parent / "hand_landmarker.task"

    # 2.配置MediaPipe手部关键点检测器
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1
    )
    # 3.打开之前测试成功的手机摄像头
    cap = cv2.VideoCapture(1,cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")
    window_name = "MediaPipe Hands"
    try:
        # 4.加载MediaPipe模型
        # 当前等待确认的候选状态
        candidate_state = None

        # 候选状态连续出现的帧数
        consecutive_frames = 0

        # 已经确认的稳定状态
        stable_state = None

        # 中指独立保存候选状态、连续帧数和稳定状态，避免影响食指判断
        middle_candidate_state = None
        middle_consecutive_frames = 0
        middle_stable_state = None
        # 只有先确认两根手指都伸直，才允许记录下一次握拳
        fist_ready = False

        # 记录本次手部持续可见期间完成的握拳次数
        fist_count = 0

        # 第一次握拳的确认时间，None表示当前尚未开始计时
        first_fist_time = None

        # 两次握拳确认之间允许的最大间隔，单位为秒
        double_fist_window = 3.0

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
            while True:
                # 5.读取一帧BGR图像
                ret, frame = cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break
                # 6.将BGR转为RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 7.将Numpy图像转为MediaPipe图像
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )
                # 8.获取当前时间戳，单位为毫秒
                timestamp_ms = int(time.monotonic() * 1000)
                # 9.检测手部关键点
                result = detector.detect_for_video(mp_image, timestamp_ms)

                # 如果当前帧没有检测到手，则中断连续计数
                if not result.hand_landmarks:
                    candidate_state = None
                    consecutive_frames = 0
                    stable_state = None  # 手丢失后清除旧状态，重新进行连续帧确认
                    # 手丢失时同时清除中指状态，重新进行连续三帧确认
                    middle_candidate_state = None
                    middle_consecutive_frames = 0
                    middle_stable_state = None
                    # 手离开画面后，需要重新伸直两指才能开始计数
                    fist_ready = False

                    # 清除当前这组动作的计数，避免离开前后的动作拼接
                    fist_count = 0
                    # 手离开画面时，同时清除本组动作的起始时间
                    first_fist_time = None
                # 1. 获取图像高度和宽度
                height, width = frame.shape[:2]

                # 2. 遍历检测到的每一只手
                for hand_landmarks in result.hand_landmarks:

                    # 3. 遍历MediaPipe预先定义的手部骨架连接关系
                    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        # connection.start：连线起点的关键点编号
                        # connection.end：连线终点的关键点编号
                        start_point = hand_landmarks[connection.start]
                        end_point = hand_landmarks[connection.end]

                        # 将起点的归一化坐标转换成像素坐标
                        x1 = int(start_point.x * width)
                        y1 = int(start_point.y * height)

                        # 将终点的归一化坐标转换成像素坐标
                        x2 = int(end_point.x * width)
                        y2 = int(end_point.y * height)

                        # 4. 绘制白色骨架连线
                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 255),
                            2
                        )

                    # 5. 绘制绿色关键点
                    for point in hand_landmarks:
                        x = int(point.x * width)
                        y = int(point.y * height)

                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    # 1. 获取食指的4个关键点
                    p5 = hand_landmarks[5]
                    p6 = hand_landmarks[6]
                    p7 = hand_landmarks[7]
                    p8 = hand_landmarks[8]

                    # 2. 将归一化坐标转换为像素坐标
                    a = (p5.x * width, p5.y * height)
                    b = (p6.x * width, p6.y * height)
                    c = (p7.x * width, p7.y * height)
                    d = (p8.x * width, p8.y * height)

                    # 3. 计算食指PIP关节的角度
                    pip_angle = calculate_angle(a, b, c)

                    # 4. 计算食指DIP关节的角度
                    dip_angle = calculate_angle(b, c, d)

                    # 1. 取出0号手腕关键点
                    p0 = hand_landmarks[0]

                    # 2. 将手腕坐标转换成像素坐标
                    wrist = (p0.x * width, p0.y * height)

                    # 3. 将食指MCP关节坐标转换成像素坐标
                    mcp = (p5.x * width, p5.y * height)

                    # 4. 将食指PIP关节坐标转换成像素坐标
                    pip = (p6.x * width, p6.y * height)

                    # 5. 计算以5号MCP为顶点的近似角度
                    mcp_angle = calculate_angle(wrist, mcp, pip)
                    # 分别判断三个角度是否满足伸直条件
                    mcp_threshold = 120
                    mcp_ok = mcp_angle > mcp_threshold
                    pip_ok = pip_angle > 160
                    dip_ok = dip_angle > 160

                    # 1. 根据三个关节角度进行当前帧判断
                    if mcp_ok and pip_ok and dip_ok:
                        raw_state = "Straight"
                    else:
                        raw_state = "Bent"
                    if raw_state == candidate_state:

                        # 状态相同，连续次数加1
                        consecutive_frames += 1

                    else:
                        # 状态不同，重新开始计数
                        candidate_state = raw_state
                        consecutive_frames = 1

                    # 3. 只有连续3帧相同，才确认状态
                    if consecutive_frames >= 3:
                        stable_state = candidate_state
                    # 4. 在摄像头左上角显示判断结果
                    # 尚未确认状态时，显示Checking
                    display_state = stable_state if stable_state is not None else "Checking"

                    # 根据两栏文字的最大宽度确定右栏位置，保持与食指相同的字号
                    angle_sample = "MCP:180.0 PIP:180.0 DIP:180.0"
                    angle_text_width = cv2.getTextSize(
                        angle_sample, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )[0][0]
                    index_text_width = cv2.getTextSize(
                        "Index Finger: Straight", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )[0][0]
                    middle_text_width = cv2.getTextSize(
                        "Middle Finger: Straight", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )[0][0]
                    right_column_x = max(
                        width // 2 + 20, 20 + max(index_text_width, angle_text_width) + 30
                    )

                    # 原始画面过窄时只在右侧补黑边，避免文字重叠或超出窗口
                    display_width = right_column_x + max(middle_text_width, angle_text_width) + 20
                    if display_width > frame.shape[1]:
                        frame = cv2.copyMakeBorder(
                            frame, 0, 0, 0, display_width - frame.shape[1],
                            cv2.BORDER_CONSTANT, value=(0, 0, 0)
                        )

                    # 在画面左栏显示稳定后的食指状态
                    cv2.putText(
                        frame,
                        f"Index Finger: {display_state}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    # 5. 显示三个关节的角度
                    cv2.putText(
                        frame,
                        f"MCP:{mcp_angle:.1f} PIP:{pip_angle:.1f} DIP:{dip_angle:.1f}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                    # 获取中指根部的像素坐标
                    middle_mcp = (
                        hand_landmarks[9].x * width,
                        hand_landmarks[9].y * height
                    )

                    # 获取中指PIP关节的像素坐标，作为夹角顶点
                    middle_pip = (
                        hand_landmarks[10].x * width,
                        hand_landmarks[10].y * height
                    )

                    # 获取中指DIP关节的像素坐标
                    middle_dip = (
                        hand_landmarks[11].x * width,
                        hand_landmarks[11].y * height
                    )

                    # 获取中指指尖的像素坐标，用于计算DIP关节角度
                    middle_tip = (
                        hand_landmarks[12].x * width,
                        hand_landmarks[12].y * height
                    )

                    # 与食指一致，计算并显示中指的MCP和DIP关节角度
                    middle_mcp_angle = calculate_angle(wrist, middle_mcp, middle_pip)
                    middle_dip_angle = calculate_angle(middle_pip, middle_dip, middle_tip)

                    # 复用现有函数，计算中指PIP关节角度
                    middle_pip_angle = calculate_angle(
                        middle_mcp, middle_pip, middle_dip
                    )

                    # 中指与食指使用相同的三个关节伸直条件
                    middle_mcp_ok = middle_mcp_angle > mcp_threshold
                    middle_pip_ok = middle_pip_angle > 160
                    middle_dip_ok = middle_dip_angle > 160

                    # 中指只有伸直和弯曲两种当前帧判断结果
                    if middle_mcp_ok and middle_pip_ok and middle_dip_ok:
                        middle_raw_state = "Straight"
                    else:
                        middle_raw_state = "Bent"

                    # 相同候选状态连续出现时累加，否则从当前帧重新计数
                    if middle_raw_state == middle_candidate_state:
                        middle_consecutive_frames += 1
                    else:
                        middle_candidate_state = middle_raw_state
                        middle_consecutive_frames = 1

                    # 连续三帧判断一致后，才更新中指的稳定状态
                    if middle_consecutive_frames >= 3:
                        middle_stable_state = middle_candidate_state

                    # 与食指一样，首次确认前只显示Checking作为等待提示
                    middle_display_state = (
                        middle_stable_state if middle_stable_state is not None else "Checking"
                    )

                    # 在右栏显示中指状态，与左栏食指的高度、字体、字号和颜色一致
                    cv2.putText(
                        frame,
                        f"Middle Finger: {middle_display_state}",
                        (right_column_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    # 中指的三个角度与食指角度显示在同一行，使用相同的字号和颜色
                    cv2.putText(
                        frame,
                        f"MCP:{middle_mcp_angle:.1f} PIP:{middle_pip_angle:.1f} DIP:{middle_dip_angle:.1f}",
                        (right_column_x, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                    # 任意一根手指尚未完成首次三帧确认时，暂时等待
                    if stable_state is None or middle_stable_state is None:
                        fist_state = "Checking"

                    # 两根手指的稳定状态都为弯曲时，判定为简化握拳
                    elif stable_state == "Bent" and middle_stable_state == "Bent":
                        fist_state = "Fist"

                    # 只要有一根手指的稳定状态为伸直，就不满足握拳条件
                    else:
                        fist_state = "Not Fist"
                    # 主程序没有进行中的会话时，才接受新的握拳唤醒
                    if is_busy is None or not is_busy():
                        # 获取本帧的单调时钟读数，单位为秒
                        current_time = time.monotonic()

                        # 已完成第一次握拳时，检查等待第二次的时间
                        if fist_count == 1 and first_fist_time is not None:
                            # 计算距离第一次确认握拳已经过去多少秒
                            elapsed_time = current_time - first_fist_time

                            # 超时后取消本组动作，需要重新伸直两指再握拳
                            if elapsed_time > double_fist_window:
                                fist_count = 0
                                first_fist_time = None
                                fist_ready = False
                                print("两次握拳间隔超时，请重新开始")

                        # 两根手指都稳定伸直后，允许记录下一次握拳
                        if stable_state == "Straight" and middle_stable_state == "Straight":
                            fist_ready = True

                        # 确认一次新的握拳动作
                        elif fist_state == "Fist" and fist_ready:
                            fist_count += 1  # 握拳次数增加一次
                            fist_ready = False  # 持续握拳不会重复计数
                            print(f"已完成握拳次数：{fist_count}")
                            # 第一次确认握拳时，记录本组动作的开始时间
                            if fist_count == 1:
                                first_fist_time = current_time

                            # 确认两次握拳后，清除本组动作，避免重复计数
                            if fist_count >= 2:
                                fist_count = 0
                                fist_ready = False
                                first_fist_time = None

                                # 调用主程序提供的会话启动函数
                                if on_wake is not None:
                                    on_wake()
                                else:
                                    print("检测到两次有效握拳")

                cv2.imshow(window_name, frame)

                # 按Q键退出
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # 点击窗口右上角的×也可以退出
                try:
                    if cv2.getWindowProperty(
                            window_name, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
                except cv2.error:
                    break

    finally:
        # 13. 释放摄像头和窗口资源
        cap.release()
        cv2.destroyAllWindows()
# 直接运行本文件时启动检测，被其他文件导入时不自动启动
if __name__ == "__main__":
    run_hand_detection()

