# 학습을 마친 두개의 YOLO 모델(8, 11)을 동일한 테스트 이미지들로 성능을 비교 평가하고, 그 예측 결과를 시각적인 이미지 파일로 각각 저장하는 것입니다. 어떤 모델이 실제 상황에서 더 나은 성능을 보일지 직접 눈으로 확인하기 위한 '실기 시험' 코드라고 할 수 있습니다.
""" 1. 라이브러리 불러오기 - "도구 챙기기"
from ultralytics import YOLO: 우리가 설치한 ultralytics 라이브러리에서, YOLO 모델을 불러오고, 학습시키고, 예측하는 데 필요한 모든 기능이 담겨있는 YOLO라는 핵심 부품(클래스)을 가져옵니다.
import os: 파일 경로를 다루는 데 사용되는 파이썬 기본 도구함입니다. 지금 당장 쓰진 않더라도, 경로 관련 작업을 할 때는 항상 챙겨두는 좋은 습관입니다. """
from ultralytics import YOLO
import os

""" 2. 경로 설정 - "실험 재료 준비하기"
이 부분은 스크립트에게 "어떤 모델을 가지고", "어떤 이미지를 분석할지" 알려주는 가장 중요한 설정 영역입니다.
yolov8_model_path: 우리가 'debris' 클래스 하나만으로 학습시킨 **YOLOv8 모델의 최종 결과물(best.pt)**이 어디에 저장되어 있는지 알려주는 경로입니다. 첫 번째 선수를 지정하는 것과 같습니다.
yolov11_model_path: 마찬가지로, 'debris' 클래스만으로 학습시킨 YOLOv11 모델의 best.pt 파일 경로입니다. 두 번째 선수를 지정합니다.
image_source_path: 두 모델이 실력을 겨룰 '시험 문제지'인 테스트 이미지들이 어디에 있는지 알려주는 경로입니다. 폴더 전체를 지정하면 그 안의 모든 이미지에 대해 예측을 수행합니다. """
# 1. 'debris' 클래스만으로 학습된 YOLOv8 모델의 'best.pt' 파일 경로
# (train_yolov8.py 실행 시 name='debris_yolov8_debris_only_run'으로 지정했던 결과)
yolov8_model_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\runs\detect\debris_yolov8_debris_only_run\weights\best.pt'

# 2. 'debris' 클래스만으로 학습된 YOLOv11 모델의 'best.pt' 파일 경로
# (train_yolov11.py 실행 시 name='debris_yolov11_debris_only_run'으로 지정했던 결과)
yolov11_model_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\runs\detect\debris_yolov11_debris_only_run\weights\best.pt'

# 3. 비교할 이미지가 있는 폴더 또는 특정 이미지 파일의 경로
# (debris_only_dataset의 test/images 폴더를 통째로 넣는 것을 추천)
image_source_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\debris_only_dataset\test\images'

# -------------------------------------------------------------------------------------------------------------------------------------------------------

""" 3. 모델 로드 - "선수들을 경기장에 불러오기"
model_v8 = YOLO(yolov8_model_path): 위에서 지정한 경로의 YOLOv8 가중치 파일(best.pt)을 실제로 불러와서, 예측을 수행할 준비가 된 model_v8 이라는 선수(객체)를 생성합니다. 이 순간, 50번의 훈련을 통해 얻은 모든 지식이 컴퓨터 메모리로 올라옵니다.
model_v11 = YOLO(yolov11_model_path): YOLOv11 선수도 똑같이 준비시킵니다. """
# 모델 로드
print("두 모델을 로드합니다...")
model_v8 = YOLO(yolov8_model_path)
model_v11 = YOLO(yolov11_model_path)
print("모델 로드 완료.")

""" 4. 예측 실행 - "본격적인 실력 대결"
if __name__ == '__main__':: 이 스크립트가 직접 실행될 때만 아래 코드를 작동시키라는 의미의 파이썬 관용구입니다. "여기서부터가 진짜 실행 부분이다"라고 생각하시면 됩니다.
model_v8.predict(...): 드디어 YOLOv8 선수에게 예측 임무를 시키는 명령입니다. 각 설정값(인자)의 의미는 다음과 같습니다.
source=image_source_path: "분석할 이미지는 image_source_path에 있는 것들을 사용해라."
save=True: "예측 결과(박스가 그려진 이미지)를 파일로 저장해라." (이게 없으면 결과가 화면에만 보이고 저장되지 않습니다.)
project='comparison_results': "결과를 저장할 때, comparison_results라는 이름의 최상위 폴더를 만들어라."
name='yolov8_predictions': "그 안에 yolov8_predictions라는 이름으로 이번 예측 결과를 저장할 폴더를 만들어라."
exist_ok=True: "만약 폴더가 이미 존재하더라도 오류를 내지 말고 그냥 덮어써라." (여러 번 실행해도 편리합니다.)
conf=0.5: "모델이 50% 이상의 확신(Confidence)을 갖는 예측 결과만 보여주고 저장해라." (불확실한 추측은 제외해서 결과를 깔끔하게 볼 수 있습니다.)
model_v11.predict(...): YOLOv11 선수에게도 똑같은 임무를 부여합니다. 단, 결과는 yolov11_predictions 라는 다른 폴더에 저장하여 서로 섞이지 않도록 합니다. """
# 예측 실행 및 결과 저장
if __name__ == '__main__':
    print("\n[YOLOv8 모델 예측 시작]")
    # YOLOv8 예측 결과 저장
    model_v8.predict(
        source = image_source_path,
        save = True,
        # imgsz = 640,
        # batch = 8,
        project = 'comparison_results', # 결과를 저장할 상위 폴더
        name = 'yolov8_predictions', # 하위 폴더
        exist_ok = True,
        conf = 0.5 # 신뢰도 50% 이상만 표시
    )
    print("[YOLOv8 모델 예측 완료]")

    print("\n[YOLOv11 모델 예측 시작]")
    # YOLOv11 예측 결과 저장
    model_v11.predict(
        source = image_source_path,
        save = True,
        # imgsz = 640,
        # batch = 8,
        project = 'comparison_results', 
        name = 'yolov11_predictions',
        exist_ok = True,
        conf = 0.5
    )
    print("[YOLOv11 모델 예측 완료]")

    """ 5. 결과 안내 - "경기 결과 발표"
    마지막 print 문들은 모든 작업이 끝났음을 알려주고, 결과가 정확히 어느 폴더에 저장되었는지 다시 한번 안내해주는 친절한 메시지입니다.
    한 문장으로 요약하자면, 이 스크립트는 두 명의 숙련된 탐정(모델)에게 똑같은 사건 현장 사진(이미지) 뭉치를 보여주고, 각자의 수사 보고서(예측 결과)를 별도의 파일 캐비닛에 정리하도록 하는 자동화된 절차입니다. """
    print("\n 모든 비교 예측이 완료되었습니다. 'comparison_results' 폴더를 확인하세요.")
    print(" - YOLOv8 결과: comparison_results/yolov8_predictions")
    print(" - YOLOv11 결과: comparison_results/yolov11_predictions")

