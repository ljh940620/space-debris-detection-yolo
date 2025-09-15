import os
import torchvision
from tqdm import tqdm
from PIL import Image
import random
# [함수정의] calculate_iou(박스1, 박스2):
def calculate_iou(box1, box2):
    # box = [x_center, y_center, width, height]
    # 두 박스의 좌표를 (중심,너비,높이)에서 (좌상단, 우하단)으로 변환
    # box1 좌상단, 우하단
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2
    # box2 좌상단, 우하단
    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2
    # 겹치는 영역의 (좌상단, 우하단) 좌표를 계산 (max, min 활용)
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    # 겹치는 영역의 넓이를 계산(음수면 0으로 처리)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    # 두 박스의 넓이를 각각 계산
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    # 합집합 영역의 넓이를 계산(박스1 넓이 + 박스2 넓이 - 겹치는 영역 넓이)
    union_area = box1_area + box2_area - inter_area
    # IoU를 계산 (겹치는 영역 넓이 / 합집합 영역 넓이)
    iou = inter_area / union_area
    # IoU 값 반환
    return iou

# 1. 설정값들 정의 (이미지 크기, 개수, 숫자 크기, IoU 한계치, 저장 경로 등)
bg_width, bg_height = 512, 128
num_samples = 10
min_scale, max_scale = 0.2, 0.3
IOU_THRESHOLD = 0.1
output_dir = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\my_mini_dataset_multi"

# 2. 결과 저장할 폴더들 생성
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# 3. MNIST 데이터 불러오기
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)

print(f"'{output_dir}' 폴더에 {num_samples}개의 샘플 생성을 시작합니다.")

# 4. 만들 이미지 개수만큼 [바깥 루프] 시작:
for i in tqdm(range(num_samples), desc="다중 숫자 데이터셋 생성 중", ncols=75, unit="개"):
    # 1. 깨끗한 흰 배경 이미지 생성
    background = Image.new('L', (bg_width, bg_height), 255)
    # 2. 이번 이미지의 라벨들을 저장할 빈 리스트 생성
    labels_for_this_image = []
    # 3. 이번 이미지에 성공적으로 배치된 박스들을 저장할 빈 리스트 생성
    placed_boxes = []
    # 4. 이번 이미지에 배치할 숫자(2~3개를) 무작위로 결정
    num_digits_to_place = random.randint(2, 3)
    #print(f"이번 이미지에 배치할 숫자 개수 : {num_digits_to_place}개")

    # 5. 성공적으로 배치된 박스 개수가 목표 개수보다 적은 동안 [중간 루프] 반복:
    while len(placed_boxes) < num_digits_to_place:
        # (안전장치) 시도 횟수 100번으로 제한, 배치 성공 여부 변수(placed) False로 초기화
        # 시도 횟수가 100번보다 적은 동안 [안쪽 루프] 반복:
        max_attempts = 100
        attempt = 0
        placed = False
        #print("성공적으로 배치된 박스 개수가 목표 개수보다 적습니다.")

        while attempt < max_attempts: 
            # 1. 랜덤 크기, 랜덤 위치로 '후보 박스' 하나 생성
            mnist_image, _ = random.choice(mnist_dataset)
            scale = random.uniform(min_scale, max_scale)
            new_size_width = int(bg_width * scale)
            new_size_height = int(bg_height * scale)
            resized_mnist = mnist_image.resize((new_size_width, new_size_height))

            max_x = bg_width - new_size_width
            max_y = bg_height - new_size_height
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            norm_x = (paste_x + new_size_width / 2) / bg_width
            norm_y = (paste_y + new_size_height / 2) / bg_height
            norm_w = new_size_width / bg_width
            norm_h = new_size_height / bg_height
            candidate_box = [norm_x, norm_y, norm_w, norm_h]
            #print(f"후보박스 생성 : {candidate_box}")
            
            # 2. '후보 박스'와 '이미 배치된 박스들'을 하나씩 비교하며 충돌 검사:
            # iou = calculate_iou(후보박스, 이미 배치된 박스)
            # 만약 iou > 한계치 이면:
            # "겹친다"고 표시하고 검사 중단
            is_overlapping = False
            for existing_box in placed_boxes:
                #print("후보박스와 배치된 박스 비교")
                iou = calculate_iou(candidate_box, existing_box)
                if iou > IOU_THRESHOLD:
                    is_overlapping = True
                    #print(f"{existing_box}은 {candidate_box}과 겹친다")
                    break
            # 3. 만약 "겹치지 않는다"고 판단되면:
            # 배경 이미지에 후보 숫자들 붙여넣기
            # 라벨 리스트에 후보 박스의 라벨 정보 추가
            # '이미 배치된 박스들' 리스트에 후보 박스 정보 추가
            # 배치 성공 여부(placed)를 True로 변경
            # [안쪽 루프] 빠져나가기 (break)
            if not is_overlapping:
                #print("겹치지 않습니다.")
                background.paste(resized_mnist, (paste_x, paste_y), resized_mnist)
                label_string = f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"
                labels_for_this_image.append(label_string)
                placed_boxes.append(candidate_box)
                placed = True
                #print(f"숫자 라벨 리스트 개수 : {len(labels_for_this_image)}개")
                #print(f"라벨 리스트 : {labels_for_this_image}")
                break
            # 4. 겹쳤다면 시도 횟수 1 증가
            attempt += 1
            #print(f"겹칩니다. 시도 : {attempt}")

        # 만약 배치에 최종 실패했다면(placed가 False이면):
        # 이번 이미지 생성을 포기하고 [중간 루프] 빠져나가기 (break)
        if not placed:
            #print("이미지 생성을 포기합니다.")
            break

        # 6. 만약 목표한 개수만큼 배치가 다 성공했다면:
        # 최종 이미지를 파일로 저장
        # 라벨 리스트에 있는 모든 라벨을 텍스트 파일 하나에 저장
        if len(placed_boxes) == num_digits_to_place:
            #print(f"배치완료, 라벨 리스트 개수: {len(labels_for_this_image)}")
            image_filename = f"{i:04d}.png"
            background.save(os.path.join(output_dir, "images", image_filename))
            label_filename = f"{i:04d}.txt"
            with open(os.path.join(output_dir, "labels", label_filename), "w") as f:
                f.write("\n".join(labels_for_this_image))

# 5. 모든 작업 완료 메시지 출력
print("다중 숫자 데이터셋 생성이 완료되었습니다!")
