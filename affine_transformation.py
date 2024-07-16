import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def calculate_bounding_box(segment_points):    
    # Initialize min and max values with the first point
    min_x, min_y = segment_points[0]
    max_x, max_y = segment_points[0]
    
    # Traverse all points to find the minimum and maximum coordinates
    for i in range(len(segment_points)):
        if segment_points[i][0] > max_x:
            max_x = segment_points[i][0]
        elif segment_points[i][0] < min_x:
            min_x = segment_points[i][0]
        elif segment_points[i][1] < min_y:
            min_y = segment_points[i][1]
        elif segment_points[i][1] > max_y:
            max_y = segment_points[i][1]
            
    # Define the four corner points of the bounding box
    top_left = (min_x, min_y)
    top_right = (max_x, min_y)
    bottom_left = (min_x, max_y)
    bottom_right = (max_x, max_y)
    
    return top_left, top_right, bottom_left, bottom_right


image_original = cv.imread(r"D:\code\20240520_190301_e4c9a8d6-47b3-4eec-b2dd-32ea058edbfd.jpg")
H, W, _ = image_original.shape
cv.imshow("image_original", image_original)

cengban_model = YOLO(r'D:\code\yolov8\runs\Segment\candy_cengban\weights\best.pt')
cengban_result = cengban_model(image_original)
huojia_model = YOLO(r'D:\code\yolov8\runs\Segment\candy_huojia\weights\best.pt')
huojia_result = huojia_model(image_original)

cengban_left = cengban_result[0].masks.xy
left = 40000
right = 0
left_y = 0
right_y = 0
for i in cengban_result[0].masks.xy[0]:
    if i[0] < left:
        left = i[0]
        left_y = i[1]
    if i[1] > right:
        right = i[0]
        right_y = i[1]
# 如果高低差超过阈值则进行透视变换
if abs(left_y - right_y) > -1:
    top_left, top_right, bottom_left, bottom_right = calculate_bounding_box(huojia_result[0].masks.xy[0])
    left_top = [top_left[1], top_left[0]]
    right_top = [top_right[1], top_right[0]]
    left_bottom = [bottom_left[1], bottom_left[0]]
    right_bottom = [bottom_right[1],bottom_right[0]]
    # 这里求的是外接矩阵，因此，是个矩形，如下仿射变换其实就是底边上移
    points1 = np.float32([left_top, right_top, left_bottom, right_bottom])
    points2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

    mat_perspective = cv.getPerspectiveTransform(points1, points2)
    image_perspective = cv.warpPerspective(image_original, mat_perspective,
                                        (image_original.shape[1], image_original.shape[0]))

    cv.imshow("image_perspective", image_perspective)
    cv.waitKey(delay=0)
    cv.destroyAllWindows()
else:
    print("不进行仿射变换")