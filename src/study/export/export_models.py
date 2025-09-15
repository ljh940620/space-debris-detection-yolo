from ultralytics import YOLO

# 1. 비교할 두 모델의 'best.pt' 파일 경로
# 'debris' 클래스만으로 학습된 YOLOv8 모델의 'best.pt' 파일 경로
# (train_yolov8.py 실행 시 name='debris_yolov8_debris_only_run'으로 지정했던 결과)
yolov8_model_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\runs\detect\debris_yolov8_debris_only_run\weights\best.pt'

# 'debris' 클래스만으로 학습된 YOLOv11 모델의 'best.pt' 파일 경로
# (train_yolov11.py 실행 시 name='debris_yolov11_debris_only_run'으로 지정했던 결과)
yolov11_model_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\runs\detect\debris_yolov11_debris_only_run\weights\best.pt'

# 1단계: 모델을 표준 형식(ONNX)으로 변환하기
# 두 모델의 내부 구조를 쉽게 들여다볼 수 있도록, 국제 표준 설계도와 같은 ONNX 형식으로 변환하겠습니다.
if __name__ == '__main__':
    print("--- [YOLOv8n 모델을 ONNX] ---")
    model_v8 = YOLO(yolov8_model_path)
    model_v8.export(format='onnx')
    print("--- [YOLOv8n.onnx 파일 생성 완료] ---\n")
    
    print("--- [YOLOv11n 모델을 ONNX 형식으로 변환 시작] ---")
    model_v11 = YOLO(yolov11_model_path)
    model_v11.export(format='onnx')
    print("--- [YOLOv11n.onnx 파일 생성 완료] ---")
    