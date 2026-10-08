import cv2

video_path = r"C:/Users/omnyx/CCTV_Project/data/videos/aihub_cctv.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video")
    exit()

print(f"Video opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
        continue

    cv2.imshow("Test", frame)
    key = cv2.waitKey(100)  # 100ms delay
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
