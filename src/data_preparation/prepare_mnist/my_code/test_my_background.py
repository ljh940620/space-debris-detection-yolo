import os
import torchvision
from tqdm import tqdm
import random
from PIL import Image
# [함수 정의] calculate_iou(박스1, 박스2)
def calculate_iou(box1, box2):
    b1x1 = box1[0] - box1[2] / 2
    b1y1 = box1[1] - box1[3] / 2
    b1x2 = box1[0] + box1[2] / 2
    b1y2 = box1[1] + box1[3] / 2
    
    b2x1 = box2[0] - box2[2] / 2
    b2y1 = box2[1] - box2[3] / 2
    b2x2 = box2[0] + box2[2] / 2
    b2y2 = box2[1] + box2[3] / 2

    inter_x1 = max(b1x1, b2x1)
    inter_y1 = max(b1y1, b2y1)
    inter_x2 = min(b1x2, b2x2)
    inter_y2 = min(b1y2, b2y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    iou = inter_area / union_area
    return iou

# 1. 설정값 
# 배경 이미지 크기
bg_width, bg_height = 512, 128
# 이미지 개수
num_samples = 20
# 숫자 크기
min_scale, max_scale = 0.2, 0.3
# IoU 한계치
IOU_THRESHOLD = 0.1
# 배경 이미지 파일 경로
background_folder = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\src\data_preparation\prepare_mnist\background_folder"
# 배경 이미지 무작위로 고를 수 있는 경로 리스트
background_path = [os.path.join(background_folder, fname) for fname in os.listdir(background_folder)]
# 결과 폴더 경로
output_dir = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\my_mini_dataset_real_bg"


# 2. 폴더 만들기 images, labels
os.makedirs(os.path.join(output_dir, "images"),exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"),exist_ok=True)

# 3. MNIST 데이터셋 다운로드
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)

# 4. 이미지 개수만큼 배경 이미지에 숫자 넣기: [바깥루프]
for i in tqdm(range(num_samples), desc="실제 배경 데이터셋 생성 중", ncols=100, unit="개"):
    # 가. 배경 이미지 무작위로 불러오고 정해진 RGB 이미지 크기로 맞추기
    random_bg = random.choice(background_path)
    background = Image.open(random_bg)
    background = background.convert('RGB')
    background = background.resize((bg_width, bg_height))
    
    num_digits_place = random.randint(1,3)
    placed_boxes = []
    label_list = []
    # 나. 불러온 이미지에 숫자 1~3개를 넣기 위한 단계: [중간루프]
    while len(placed_boxes) < num_digits_place:
        max_attempts = 100
        attempt = 0
        placed = False
        is_overlapping = False
        # 다. 시도 횟수가 100번으로 제한(안전장치): [안쪽루프]
        while attempt < max_attempts:
            # 1. 숫자를 데이터셋에서 무작위로 하나 뽑고 숫자의 크기 및 좌표 계산 후 후보박스 하나 생성
            mnist_image, _ = random.choice(mnist_dataset)
            scale = random.uniform(min_scale, max_scale)
            #new_size_width = int(scale * bg_width)
            #new_size_height = int(scale * bg_height)
            new_size = int(scale * bg_height)
            resized_mnist = mnist_image.resize((new_size, new_size))
            
            max_x = bg_width - new_size
            max_y = bg_height - new_size
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            norm_x = (paste_x + new_size / 2) / bg_width
            norm_y = (paste_y + new_size / 2) / bg_height
            norm_w = new_size / bg_width
            norm_h = new_size / bg_height
            candidate_box = [norm_x, norm_y, norm_w, norm_h]
            # 2. 후보박스와 배치된 박스들 겹치는지 비교하며 충돌 검사: -> 겹치면 검사 중단
            for existing_box in placed_boxes:
                iou = calculate_iou(existing_box, candidate_box)
                if iou > IOU_THRESHOLD:
                    is_overlapping = True
                    break
            # 3. "안겹쳤다" -> [안쪽 루프] 빠져나오기
            if not is_overlapping:
                background.paste(resized_mnist, (paste_x, paste_y), resized_mnist)
                placed_boxes.append(candidate_box)
                placed = True
                label_string = f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"        
                label_list.append(label_string)
                break
            # 4. 겹쳤으면 시도 1회 증가
            attempt += 1
        
        # 5. 100번 시도해도 배치 가능한 숫자가 없으면 -> [중간루프] 빠져나오기
        if not placed:
            break

        # 5. 숫자 이미지가 다 배치완료 됬을 때
        if len(placed_boxes) == num_digits_place:
            image_filename = f"{i:04d}.png"
            background.save(os.path.join(output_dir, "images", image_filename))
            label_filename = f"{i:04d}.txt"
            with open(os.path.join(output_dir, "labels", label_filename), 'w') as f:
                f.write("\n".join(label_list))

print("실제 배경 데이터셋 생성이 완료되었습니다!")

