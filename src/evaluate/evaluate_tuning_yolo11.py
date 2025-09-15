from ultralytics import YOLO
import os

yaml_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\debris_only_dataset\data_debris_only.yaml'

# 1. 'debris' 클래스만으로 학습된 YOLOv8 모델의 'best.pt' 파일 경로
# (train_yolov8.py 실행 시 name='debris_yolov8_debris_only_run'으로 지정했던 결과)
yolov8_model_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\runs\detect\debris_yolov8_debris_only_run\weights\best.pt'

# 2. 'debris' 클래스만으로 학습된 YOLOv11 모델의 'best.pt' 파일 경로
# (train_yolov11.py 실행 시 name='debris_yolov11_debris_only_run'으로 지정했던 결과)
yolov11_model_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\runs\detect\debris_yolov11_debris_only_run\weights\best.pt'

# 3. 비교할 이미지가 있는 폴더 또는 특정 이미지 파일의 경로
# (debris_only_dataset의 test/images 폴더를 통째로 넣는 것을 추천)
image_source_path = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\debris_only_dataset\test\images'

if __name__ == '__main__':
    """     # --- YOLOv8 모델 최종 평가 ---
    print("--- [YOLOv8n 최종 성능 평가 시작] ---")
    model_v8 = YOLO(yolov8_model_path)
    model_v8.val(data=yaml_path, split='test', project='tuning_results', name='yolov8_default')
    #model_v8.val(data=yaml_path, split='test', iou=0.5, project='tuning_results', name='iou_0.5_yolov8')
    print("--- [YOLOv8n 최종 성능 평가 완료] ---\n") """

    # --- YOLOv11 모델 최종 평가 ---
    print("--- [YOLOv11n 최종 성능 평가 시작] ---")
    model_v11 = YOLO(yolov11_model_path)
    #model_v11.val(data=yaml_path, split='test', project='tuning_results', name='yolov11_default')
    model_v11.val(data=yaml_path, split='test', conf=0.25, project='tuning_results', name='conf_0.25')
    #model_v11.val(data=yaml_path, split='test', conf=0.25, project='tuning_results', name='conf_0.25_default')
    #model_v11.val(data=yaml_path, split='test', conf=0.5, project='tuning_results', name='conf_0.5')
    
    #model_v11.val(data=yaml_path, split='test', iou=0.7, project='tuning_results', name='iou_0.7')
    #model_v11.val(data=yaml_path, split='test', iou=0.4, project='tuning_results', name='iou_0.4')
    #model_v11.val(data=yaml_path, split='test', iou=0.35, project='tuning_results', name='iou_0.35')
    
    print("--- [YOLOv11n 최종 성능 평가 완료] ---") 