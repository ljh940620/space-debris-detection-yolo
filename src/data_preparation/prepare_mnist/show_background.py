import os
import random
from PIL import Image

# 1. 설정
background_folder = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\src\data_preparation\prepare_mnist\background_folder"
target_width, target_height = 512, 128

# 2. 경로 리스트 만들기
background_image_paths = [os.path.join(background_folder, fname) for fname in os.listdir(background_folder)]
print(f"찾은 배경 이미지들: {background_image_paths}")

# 3. 무작위로 하나 선택해서 열기
random_path = random.choice(background_image_paths)
print(f"선택된 이미지: {random_path}")
img = Image.open(random_path)

# 4. 변환 및 리사이즈
img_rgb = img.convert('RGB')
img_resized = img_rgb.resize((target_width, target_height))

# 5. 결과 확인
img_resized.save("test_output.png") # 결과물을 파일로 저장해서 확인
img_resized.show() # 이미지를 직접 화면에 띄워서 확인