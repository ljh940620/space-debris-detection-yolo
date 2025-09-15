import os
import random
import torchvision
from PIL import Image
from tqdm import tqdm
# (새로운 기능) IoU 계산 함수
# 이 문제를 해결하려면, 새로운 숫자를 배치하기 전에 "이 자리에 이미 다른 숫자가 있는가?"를 확인하는 충돌 감지(Collision Detection) 로직을 추가해야 합니다. 이 '겹침'의 정도를 IoU(Intersection over Union)라는 지표로 계산합니다.
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

# 1. 설정값 정하기
# 배경 이미지 크기
bg_width, bg_height = 512, 128

# 생성할 이미지 개수
num_samples = 10

# MNIST 숫자의 최소/최대 크기(배경 크기 대비 비율)
min_scale, max_scale = 0.2, 0.3
IOU_THRESHOLD = 0.05  # 허용할 IoU 최대값 (0에 가까울수록 안 겹침)

# 저장할 데이터셋 경로
output_dir = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\mini_dataset_multi"

# 2. 저장할 폴더 만들기
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# 3. MNIST 데이터 불러오기
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)

print(f"'{output_dir}' 폴더에 {num_samples}개의 샘플 생성을 시작합니다.")

for i in tqdm(range(num_samples), desc="다중 숫자 데이터셋 생성 중", ncols=75, unit="개"):
    background = Image.new('L', (bg_width, bg_height), 255)

    # (변경점) 이 이미지에 대한 모든 라벨을 저장할 리스트를 생성합니다.
    labels_for_this_image = []

    # (변경점) 이미지 당 2개 또는 3개의 숫자를 배치합니다.
    placed_boxes = []
    num_digits_to_place = random.randint(2, 3)

    # (변경점) while 루프로 성공할 때까지 시도
    while len(placed_boxes) < num_digits_to_place:

        # 무한 루프 방지를 위한 안전장치
        max_attempts = 100
        attempt = 0
        placed = False

        while attempt < max_attempts:
            # 1. 새로운 후보(candidate) 박스 생성
            mnist_image, _ = random.choice(mnist_dataset)
            scale = random.uniform(min_scale, max_scale)
            new_size_width = int(bg_width * scale)
            new_size_height = int(bg_height * scale)
            resized_mnist = mnist_image.resize((new_size_width, new_size_height))
            
            paste_x = random.randint(0, bg_width - new_size_width)
            paste_y = random.randint(0, bg_height - new_size_height)

            # 정규화된 좌표로 후보 박스 정의
            norm_x = (paste_x + new_size_width / 2) / bg_width 
            norm_y = (paste_y + new_size_height / 2) / bg_height
            norm_w = new_size_width / bg_width
            norm_h = new_size_height / bg_height
            candidate_box = [norm_x, norm_y, norm_w, norm_h]

            # 2. 충돌 검사
            is_overlapping = False
            for existing_box in placed_boxes:
                iou = calculate_iou(candidate_box, existing_box)
                if iou > IOU_THRESHOLD:
                    is_overlapping = True
                    break # 하나라도 겹치면 더 이상 검사할 필요 없음
            
            # 3. 배치 결정
            if not is_overlapping:
                # 겹치지 않으면 이미지에 붙이고, 리스트에 추가
                background.paste(resized_mnist, (paste_x, paste_y), resized_mnist)
                
                label_string = f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"
                labels_for_this_image.append(label_string)
                placed_boxes.append(candidate_box)
                placed = True
                break # 배치에 성공했으므로 시도 중단

            attempt += 1 # 겹쳤으면 시도 횟수 증가
        
        if not placed:
            # 100번 시도해도 빈 자리를 못 찾으면 그냥 이번 이미지는 포기하고 넘어감
            break

        # 이미지 및 라벨 저장
        if len(placed_boxes) == num_digits_to_place: # 성공적으로 다 배치했을 때만 저장
            image_filename = f"{i:04d}.png"
            background.save(os.path.join(output_dir, "images", image_filename))

            label_filename = f"{i:04d}.txt"
            with open(os.path.join(output_dir, "labels", label_filename), 'w') as f:
                f.write("\n".join(labels_for_this_image))