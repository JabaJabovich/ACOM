import cv2
import numpy as np
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