import cv2
import numpy as np

#Задание 1-3
def cam_show():
    cap = cv2.VideoCapture(0)
    lower_red1 = np.array([0, 120, 75])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([140, 115, 75])
    upper_red2 = np.array([180, 255, 255])
    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        final_mask = cv2.addWeighted(mask1,0.5, mask2, 0.5, 0.0)
        red_filtered_frame = cv2.bitwise_and(frame, frame, mask=final_mask)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        erosion = cv2.erode(final_mask, kernel, iterations=1)
        dilation = cv2.dilate(final_mask, kernel, iterations=1)

        cv2.imshow("Erosion", erosion)
        cv2.imshow("Dilation", dilation)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)
        cv2.imshow('Red Filtered Image', red_filtered_frame)
        cv2.imshow('orig', frame)
        cv2.imshow('hsv', hsv)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

cam_show()

def cam_show():
    cap = cv2.VideoCapture(0)
    lower_red1 = np.array([0, 125, 85])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([167, 115, 75])
    upper_red2 = np.array([180, 255, 255])
    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        final_mask = cv2.addWeighted(mask1,0.5, mask2, 0.5, 0.0)
        red_filtered_frame = cv2.bitwise_and(frame, frame, mask=final_mask)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Создаем список для объединенных контуров
        # merged_contours = []
        #
        # # Параметр для определения расстояния, при котором контуры будут объединяться
        # merge_threshold = 50
        #
        # Объединяем контуры
        for contour in contours:
        #     if len(merged_contours) == 0:
        #         merged_contours.append(contour)
        #     else:
        #         # Проверяем расстояние между текущим контуром и последним объединенным контуром
        #         last_contour = merged_contours[-1]
        #         x1, y1, w1, h1 = cv2.boundingRect(last_contour)
        #         x2, y2, w2, h2 = cv2.boundingRect(contour)
        #
        #         # Если расстояние между контурами меньше порога, объединяем
        #         if abs(x1 + w1 - x2) < merge_threshold:  # Проверка горизонтального расстояния
        #             merged_contours[-1] = np.vstack((merged_contours[-1], contour))
        #         else:
        #             merged_contours.append(contour)
        #
        # # Теперь создаем общий контур из всех объединенных контуров
        # if len(merged_contours) > 0:
        #     all_contours = np.vstack(merged_contours)
        #     final_merged_contour = cv2.convexHull(all_contours)

            # Рисуем объединенный контур на изображении
            if cv2.contourArea(contour) > 300:  # Если площадь контура больше 300
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 5)

        cv2.imshow('Red Filtered Image', red_filtered_frame)
        cv2.imshow('orig', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


cam_show()



