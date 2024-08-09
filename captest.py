import cv2
import time
import datetime
from threading import Thread, Lock

# 设置摄像头分辨率
frame_width = 640
frame_height = 480

# 初始化摄像头
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# 设置视频编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 计算开始时间
start_time = time.time()

# 创建帧缓存列表
frames = []

frames_lock = Lock()

# 定义一个标志来控制子线程
stop_thread = False

# 定义保存视频的函数
def save_video(frames):
    if frames:
        # 生成文件名，格式为 yyyy-mm-dd_hh-mm-ss.avi
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
        out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

        # 使用线程锁访问frames
        # 这里不要对frames重复加锁,否则会造成死锁
        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved as {filename}")

# 定义用于子线程执行的线程函数
def save_video_periodically(start_time):
    global stop_thread
    while not stop_thread:
        try:
            # 每隔1分钟保存一次视频
            if time.time() - start_time >= 60:
                with frames_lock:
                    # 保存视频流
                    save_video(frames)
                    print('one times')
                    # 清理帧缓存列表
                    frames.clear()
                    # 重新计时
                    start_time = time.time()

        except Exception as e:
            print(f"Error in save_video_thread: {e}")

# 创建视频流保存子线程
# python的子线程不同于C++的子线程;Python子线程的承接类型是函数;C++子线程的承接类型是类对象
# 因此这就决定了Python的子线程只是一个短暂的任务,必须在子线程中加入while;否则子线程不会一直运行
save_video_thread = Thread(target=save_video_periodically, args=(start_time,))
save_video_thread.daemon = True
save_video_thread.start()

try:
    while True:
        # 读取一帧数据
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧添加到帧列表中
        # 必须考虑给frames加上线程锁,因为frames是共享资源；不加锁会造成多线程竞争资源造成死锁
        with frames_lock:
            frames.append(frame)
            # 显示视频
            cv2.imshow('frame', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error Main Loop: {e}")

finally:
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    # 回收子线程
    stop_thread = True
    save_video_thread.join()