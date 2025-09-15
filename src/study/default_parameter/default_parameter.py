# ultralytics 라이브러리가 설치되어 있어야 합니다.
# pip install ultralytics

# ultralytics 라이브러리의 기본 설정을 담고 있는 DEFAULT_CFG를 직접 임포트합니다.
try:
    from ultralytics.cfg import DEFAULT_CFG

    # DEFAULT_CFG 객체에서 conf와 iou 값을 가져옵니다.
    default_conf = DEFAULT_CFG.conf
    default_iou = DEFAULT_CFG.iou

    print(f"YOLO 모델의 기본 Confidence Threshold (conf) 값: {default_conf}")
    print(f"YOLO 모델의 기본 NMS IoU Threshold (iou) 값: {default_iou}")

except ImportError:
    print("오류: 'ultralytics.cfg'에서 DEFAULT_CFG를 찾을 수 없습니다.")
    print("라이브러리 버전이 매우 오래되었거나, 설치에 문제가 있을 수 있습니다.")
    print("터미널에서 'pip install --upgrade ultralytics'로 라이브러리를 최신 버전으로 업데이트해 보세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")