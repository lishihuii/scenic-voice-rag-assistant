import math
from tts_module import play_audio  # 导入刚添加的音频播放函数

play_audio(r"/rag/test_tool/welcome.mp3")
print("播放函数已经返回")
# def calculate_angle(a,b,c):
#     # a,b,c分别代表三个点的像素坐标
#     # b是我们需要计算夹角的顶点
#     # 1.计算从B指向A的方向角
#     angle1 = math.atan2(a[1] - b[1], a[0] - b[0])
#     # 2.计算从B指向C的方向角
#     angle2 = math.atan2(c[1] - b[1], c[0] - b[0])
#
#     # 3.两个方向角相减，并转换为角度
#     angle = abs(math.degrees(angle1 - angle2))
#     # 4.保证最终夹角在0-180度之间
#     if angle >180:
#         angle = 360 - angle
#     # 5.最终返回结果
#     return angle
# # 情况一：A、B、C在一条直线上
# angle1 = calculate_angle(
#     (0, 0),
#     (3, 0),
#     (6, 0)
# )
#
# # 情况二：A、B、C形成一个直角
# angle2 = calculate_angle((0, 0), (2, 0), (2, 2))
#
# print("情况一：", angle1)
# print("情况二：", angle2)