import cv2
import numpy as np
# Задание 1-2
img1 = cv2.imread('POE2.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('POE2.jpg',cv2.IMREAD_REDUCED_COLOR_2)
img3 = cv2.imread('POE2.jpg',cv2.IMREAD_ANYCOLOR)
cv2.namedWindow('Display1',cv2.WINDOW_NORMAL)
cv2.namedWindow('Display2',cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Display3',cv2.WINDOW_AUTOSIZE)
cv2.imshow('Display3',img2)
cv2.imshow('Display2',img3)
cv2.imshow('Display1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Задание 3
cap = cv2.VideoCapture('bu-ispugalsya-ne-boysya-ya-drug.mp4',cv2.CAP_ANY)
while True:
    ret, frame = cap.read()
    if not(ret):
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)
    cv2.imshow('frame', frame)
    delay = 16
    if cv2.waitKey(delay) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

# Задание 4
cap = cv2.VideoCapture("bu-ispugalsya-ne-boysya-ya-drug.mp4",cv2.CAP_ANY)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("output.mp4", fourcc, 60, (width, height))
while True:
    ret, frame = cap.read()
    if not(ret):
        break
    cv2.imshow('frame',frame)
    video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

# Задание 5
img = cv2.imread("POE2.jpg",cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('original_frame',img)
cv2.imshow('hsv_frame',hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Задание 6
cap=cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))/2
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2
# size = 100
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    if not ret:
        break
    # Определяем точки для креста
    cross_image = np.zeros((height, width, 3), dtype=np.uint8)
    vertical_line_width = 60
    vertical_line_height = 300
    cv2.rectangle(cross_image,
                  (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                  (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                  (0, 0, 255), 2)

    horizontal_line_width = 250
    horizontal_line_height = 55
    cv2.rectangle(cross_image,
                  (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                  (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                  (0, 0, 255), 2)

    result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

    cv2.imshow("Red Cross", result_frame)



    # points1 = np.array([[width, height - size],
    #     [width - size * np.sqrt(3)/2, height + size/2],
    #     [width + size * np.sqrt(3)/2, height + size/2]], np.int32)
    # points2 = np.array([[width, height + size],
    #     [width - size * np.sqrt(3)/2, height - size/2],
    #     [width + size * np.sqrt(3)/2, height - size/2]], np.int32)
    # cv2.polylines(frame,[points1, points2], isClosed=True, color=(255, 0, 0), thickness=5)
    #
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#Задание 7
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('motion_video.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

#Задание 8

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cross_image = np.zeros((height, width, 3), dtype=np.uint8)

    vertical_line_width = 60
    vertical_line_height = 300
    cv2.rectangle(cross_image,
                  (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                  (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                  (0, 0, 255), 2)
    rect_start_v = (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2)
    rect_end_v = (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2)

    horizontal_line_width = 250
    horizontal_line_height = 55
    cv2.rectangle(cross_image,
                  (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                  (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                  (0, 0, 255), 2)

    rect_start_h = (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2)
    rect_end_h = (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2)

    central_pixel_color = frame[height // 2, width // 2]

    color_distances = [
        np.linalg.norm(central_pixel_color - np.array([0, 0, 255])),
        np.linalg.norm(central_pixel_color - np.array([0, 255, 0])),
        np.linalg.norm(central_pixel_color - np.array([255, 0, 0]))
    ]

    closest_color_index = np.argmin(color_distances)

    if closest_color_index == 0:
        cv2.rectangle(cross_image, rect_start_h, rect_end_h, (0, 0, 255), -1)
    elif closest_color_index == 1:
        cv2.rectangle(cross_image, rect_start_h, rect_end_h, (0, 255, 0), -1)
    else:
        cv2.rectangle(cross_image, rect_start_h, rect_end_h, (255, 0, 0), -1)

    if closest_color_index == 0:
        cv2.rectangle(cross_image, rect_start_v, rect_end_v, (0, 0, 255), -1)
    elif closest_color_index == 1:
        cv2.rectangle(cross_image, rect_start_v, rect_end_v, (0, 255, 0), -1)
    else:
        cv2.rectangle(cross_image, rect_start_v, rect_end_v, (255, 0, 0), -1)

    result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

    cv2.imshow('frame', result_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#Задание 9
# cap=cv2.VideoCapture("rtsp://192.168.0.108:8080/h264_pcm.sdp")
# while True:
#     ret, frame = cap.read()
#     if not(ret):
#         break
#     cv2.imshow('mp4',frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break