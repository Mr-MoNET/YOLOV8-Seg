import cv2

def main():
    # 打开外置摄像头 (通常外置摄像头的索引是1，如果是0，请改为0)
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("无法读取帧")
            break

        # 在窗体上显示这一帧
        cv2.imshow('Camera Frame', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗体
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
