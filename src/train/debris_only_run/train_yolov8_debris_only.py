from ultralytics import YOLO

# --- 모델을 'yolov8n.pt' 사용
model = YOLO(r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\model\yolov8n.pt')

# data.yaml 파일의 경로
yaml_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\debris_only_dataset\data_debris_only.yaml'

# 모델 학습 시작
if __name__ == '__main__':
    results = model.train(
        data = yaml_path,
        epochs = 50,
        imgsz = 640,
        batch = 8,
        # 결과가 겹치지 않게 새로운 폴더 이름으로 저장
        name = 'debris_yolov8_debris_only_run'
    )
    print("YOLOv8 모델 학습이 완료되었습니다!")