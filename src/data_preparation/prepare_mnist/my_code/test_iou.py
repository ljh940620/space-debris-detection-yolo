def calculate_iou(box1, box2):
    """
    두 개의 바운딩 박스(x_center, y_center, width, height)의 IoU를 계산합니다.
    """
    # box = [x_center, y_center, w, h]
    # 중심 좌표를 좌상단, 우하단 좌표로 변환 (x1, y1, x2, y2)
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # 겹치는 영역(intersection)의 좌표
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # 겹치는 영역의 넓이 계산
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    if inter_area == 0:
        return 0.0
    
    # 각 박스의 넓이 계산
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]

    # 합집합(union) 영역의 넓이 계산
    union_area = b1_area + b2_area - inter_area

    # IoU 계산
    iou = inter_area / union_area
    return iou

box_A = [0.4, 0.4, 0.3, 0.3]
box_B = [0.65, 0.55, 0.3, 0.3]
iou = calculate_iou(box_A, box_B)
print(iou) 