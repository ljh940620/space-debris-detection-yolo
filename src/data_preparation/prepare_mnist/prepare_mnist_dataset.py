# 이 코드를 실행하고 나면, 다음 단계인 모델 훈련에 즉시 사용할 수 있는 완벽하게 준비된 이미지와 정답(라벨) 데이터셋이 완성됩니다.
import torchvision
import os
from PIL import Image
import random
from tqdm import tqdm
# --- 1. 설정값 정의 ---
# 저장할 데이터셋 경로
output_dir = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\mini_dataset"
# 생성할 이미지 개수
num_samples = 1000
# 배경 이미지 크기
bg_width, bg_height = 256, 256
# MNIST 숫자의 최소/최대 크기 (배경 크기 대비 비율)
min_scale, max_scale = 0.2, 0.5

# --- 2. MNIST 데이터셋 로딩 ---
# 훈련 데이터를 불러옵니다. 한번만 다운로드 받습니다.
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
# torch, torchvision: 파이토치(PyTorch) 관련 라이브러리입니다. 여기서는 특히 torchvision을 사용하여 MNIST라는 유명한 숫자 이미지 데이터셋을 쉽게 불러오는 데 사용합니다.
# torchvision.datasets.MNIST(...) : torchvision을 사용하여 MNIST라는 숫자 이미지 데이터셋을 불러오는 데 사용합니다.
# root="./data" : MNIST 데이터를 ./data 라는 폴더에 다운로드받거나, 이미 있다면 거기서 읽어오라는 의미입니다.
# train=True : 훈련용(training) 데이터셋을 가져오라고 지정합니다. (보통 60,000개)
# download=True : 만약 ./data 폴더에 데이터가 없으면, 인터넷에서 자동으로 다운로드하라는 의미입니다.
# --- 3. 폴더 생성 ---
# 이미지를 저장할 'images' 폴더와 라벨을 저장할 'labels' 폴더를 생성합니다.
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
# import os : 운영체제(OS)와 상호작용하는 기능을 제공합니다. 이 코드에서는 폴더(디렉토리)를 만드는 데 사용됩니다 (os.makedirs).
# os.path.join(output_dir, "images") : "mini_dataset"과 "images"를 합쳐 "mini_dataset/images"라는 경로 문자열을 만듭니다.
# os.makedirs(...) : 위에서 만든 경로에 해당하는 폴더를 생성합니다.
# exist_ok=Ture : 만약 폴더가 이미 존재하더라도 오류를 내지 말고 그냥 넘어가라는 아주 유용한 옵션입니다. 이 스크립트를 여러 번 실행해도 에러가 나지 않습니다.

print(f"'{output_dir}' 폴더에 {num_samples}개의 샘플 생성을 시작합니다.")

# --- 4. 데이터셋 생성 루프 ---
for i in tqdm(range(num_samples), desc="미니 데이터셋 생성 중", ncols=75, unit="개"):
    # --- Action Plan: 배경 이미지 생성 --
    # 순백색의 빈 캔버스를 만듭니다. 'L'은 8비트 흑백 이미지를 의미합니다.
    background = Image.new('L', (bg_width, bg_height), 255)
    # from PIL import Image : Pillow 라이브러리에서 Image 모듈을 가져옵니다. Pillow는 파이썬에서 이미지를 다루는 가장 기본적인 도구입니다. 새 이미지를 만들거나, 크기를 조절하거나, 다른 이미지를 붙여넣는 등 모든 이미지 조작을 담당합니다.
    # Image.new() : Pillow를 사용해 새로운 이미지를 만듭니다.
    # 'L' : 이미지 모드를 'L'(Luminance)로 설정합니다. 이는 0(검정)~255(흰색) 사이의 값만 갖는 8비트 흑백 이미지를 의미합니다.
    # (bg_width, bg_height) : 이미지 크기를 앞에서 설정한 256x256으로 지정합니다.
    # 255 : 이미지의 기본 색상을 255(순백색)로 채웁니다.

    # --- Action Plan: MNIST 데이터 로딩 및 무작위 선택 ---
    # MNIST 데이터셋에서 무작위로 숫자 이미지 하나를 가져옵니다.
    mnist_image, mnist_label = random.choice(mnist_dataset)
    # import random : 무작위(랜덤) 작업을 위한 라이브러리인 random을 사용해서 어떤 숫자를 고를지, 어디에 붙일지, 얼마나 크게 만들지 등을 무작위로 결정하는 데 사용됩니다.
    # random.choice(mnist_dataset) : mnist_dataset 에 들어있는 60,000개의 (이미지, 라벨) 쌍 중에서 하나를 무작위로 뽑습니다.
    # mnist_image 에는 숫자 이미지 데이터(Pillow 이미지 객체)가, mnist_label 에는 해당 숫자가 무엇인지(예: 7)가 저장됩니다. (이 코드에서는 mnist_label 을 직접 사용하지는 않습니다.)

    # --- Action Plan: 무작위 위치 및 크기 결정 ---
    # 숫자의 크기를 무작위로 결정합니다.
    scale = random.uniform(min_scale, max_scale)
    new_size = int(bg_width * scale)
    # random.uniform(0.2, 0.5) : 0.2와 0.5 사이의 소수점 값을 무작위로 뽑습니다.
    # new_size = ... : 배경 너비(256)에는 방금 뽑은 비율(scale)을 곱해 새로운 크기를 계산합니다.

    # 원본 비율을 유지하며 크기를 조절합니다.
    resized_mnist = mnist_image.resize((new_size, new_size))
    # mnist_image.resize(...) : 원본 MNIST 이미지를 방금 계산한 new_size 크기로 조절합니다.

    # 숫자를 붙여넣을 위치(좌상단 x, y 좌표)를 무작위로 결정합니다.
    # 이미지가 배경 밖으로 나가지 않도록 최대 위치를 제한합니다.
    max_x = bg_width - new_size
    max_y = bg_height - new_size
    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)
    # max_x, max_y : 숫자를 붙일 수 있는 최대 x,y 좌표를 계산합니다. 만약 숫자를 너무 오른쪽에 붙이면 이미지 밖으로 빠져나가기 때문에 이를 방지하기 위한 안전장치입니다.
    # random.randint(...) : 0부터 계산된 최대값 사이에서 정수 좌표(x,y)를 무작위로 뽑습니다. 이 좌표는 이미지를 붙여넣기 시작할 좌측 상단 모서리 지점입니다.

    # --- Action Plan: 이미지 합성 ---
    # 배경 이미지에 리사이즈된 MNIST 숫자 이미지를 붙여넣습니다.
    # 세 번째 인자는 마스크로, 숫자 모양 그대로 붙여넣기 위해 사용됩니다.
    background.paste(resized_mnist, (paste_x, paste_y), resized_mnist)
    # background.paste(...): 이미지 합성의 핵심입니다. backgroun 이미지 위에 다른 이미지를 붙여넣습니다.
    # resized_mnist (첫 번째 인자) : 무엇을 붙일지 (소스 이미지)
    # (paste_x, paste_y) (두 번째 인자) : 어디에 붙일지 (좌표)
    # resized_mnist (세 번째 인자) : 마스크(mask) 역할. 이 인자가 매우 중요합니다. MNIST 이미지는 검은 숫자와 투명한(또는 흰) 배경으로 이루어져 있는데, 마스크를 지정하면 숫자가 있는 부분(0이 아닌 값)만 붙여놓고 나머지 투명한 부분은 무시합니다. 그래서 숫자 모양 그대로 깔끔하게 붙여넣어집니다.
    
    # --- Action Plan: 이미지 저장 ---
    # 생성된 이미지를 'images' 폴더에 저장합니다. 파일명은 0000, 0001 형식으로 지정합니다.
    image_filename = f"{i:04d}.png"
    background.save(os.path.join(output_dir, "images", image_filename))
    # f"{i:04d}.png": 파일 이름은 000.png, 001.png, ..., 099.png 와 같이 세 자리 숫자로 예쁘게 포맷팅합니다.
    # background.save(...) : 최종적으로 합성된 이미지를 mini_dataset/images 폴더 안에 해당 파일 이름으로 저장합니다.

    # --- Action Plan: 라벨 저장 ---
    # 바운딩 박스 좌표 [x, y, w, h]를 계산합니다.
    # x, y는 중심 좌표, w, h는 너비와 높이입니다. YOLO 포맷은 보통 0~1 사이로 정규화합니다.
    norm_x = (paste_x + new_size / 2) / bg_width
    norm_y = (paste_y + new_size / 2) / bg_height
    norm_w = new_size / bg_width
    norm_h = new_size / bg_height
    # 좌표 정규화(Normalization): 딥러닝 모델이 더 쉽게 학습할 수 있도록, 모든 좌표를 0과 1 사이의 값으로 바꿔주는 과정입니다.
    # paste_x + new_size / 2 : 바운딩 박스의 중심 x 좌표를 계산합니다.
    # ... / bg_width : 위에서 구한 중심 x 좌표를 전체 이미지 너비로 나누어 0~1 사이의 비율 값으로 만듭니다. y, w, h도 마찬가지입니다.
    
    # 라벨 파일에 "클래스_인덱스 x y w h" 형식으로 저장합니다.
    # 여기서는 숫자가 단일 클래스이므로 클래스 인덱스는 항상 0입니다.
    label_filename = f"{i:04d}.txt"
    with open(os.path.join(output_dir, "labels", label_filename), 'w') as f:
        f.write(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    # with open(...) as f: : mini_dataset/labels 폴더 안에 000.txt 같은 이름으로 새 텍스트 파일을 엽니다.
    # f.write(...) : 파일 안에 정해진 형식으로 라벨 정보를 씁니다.
    # 0 : 객체의 클래스 번호. 여기선 '숫자' 단일 클래스이므로 항상 0입니다.
    # {norm_x:.6f} : 정규화된 좌표들을 소수점 6자리까지 형식에 맞춰 기록합니다.

    """ # 진행 상황 출력
    if (i + 1) % 100 == 0:
        print(f"    {i + 1}/{num_samples} 샘플 생성 완료...") """
    
# 루프가 모두 끝나면 자동으로 완료되므로, 완료 메시지를 출력합니다.
print("데이터셋 생성이 완료되었습니다!")    


    

