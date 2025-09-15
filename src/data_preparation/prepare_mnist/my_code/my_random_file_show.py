from PIL import Image
import os
import random
import matplotlib.pyplot as plt
# 1. 설정값
background_folder = r"C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\src\data_preparation\prepare_mnist\background_folder"
# 2. 경로 리스트 만들기
backgorund_path = [os.path.join(background_folder, fname) for fname in os.listdir(background_folder)]
# 3. 무작위로 하나 선택해서 열기
random_bg = random.choice(backgorund_path)
background = Image.open(random_bg)
# 4. 변환 및 리사이즈
background = background.convert('RGB')
background = background.resize((512, 128))
i = 0
image_filename = f"{i:03d}.png"
background.save(os.path.join(background_folder,image_filename))

plt.imshow(background) # 이미지를 화면에 올리기
plt.axis('off') # 축 정보(눈금) 끄기 (선택 사항)
plt.show() # 최종적으로 화면에 표시