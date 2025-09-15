import torch.nn as nn
# [클래스 정의] MyMiniDetector (딥러닝 모델의 기본 설계도를 상속)
class MyMiniDetector(nn.Module):
    def __init__(self): # 파이썬 클래스의 생성자(initializer) 메소드입니다.
        # 부모 클래스인 nn.Module 의 생성자를 먼저 호출하는 코드입니다. nn.Module이 가진 필수 기능들을 초기화하기 위해 반드시 가장 먼저 실행해야 하는 약속과도 같습니다.
        super(MyMiniDetector, self).__init__()
        # "백본(Backbone)"이라는 이름의 순차적인 부품 그룹을 정의:
        self.backbone = nn.Sequential(
            # (입력 데이터 형태: 이미지 여러 장, 흑백 1채널, 높이 256, 너비 256)
            # [1단계 합성곱 블록]
            # --- Block 1 ---
            # 3x3 합성곱 8개를 사용해 이미지 특징 추출 (입력 채널: 1, 출력 채널: 8)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            # ReLU(Rectified Linear Unit) 활성화 함수로 특징을 명확하게 만듦
            nn.ReLU(),
            # 2x2 맥스 풀링으로 특징 맵의 크기를 절반으로 줄임 (계산 효율성 증가)
            # 데이터 형태 변화: ... 8채널, 높이 128, 너비 128)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # --- Block 2 ---
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 헤드 정의
        # 백본의 최종 출력 크기: (B, 16, 64, 65)
        #self.head_in_features = 16 * 64 * 64
        self.head = nn.Sequential(
            # 다차원 텐서를 1차원으로 쭉 펼쳐주는 역할을 합니다. (B, 16, 64, 64) -> (B, 16 * 64 * 64)
            nn.Flatten(),
            # 완전 연결 계층(Fully Connected Layer), 모든 입력 노드가 모든 출력 노드에 연결된 가장 기본적인 신경망 레이어입니다.
            # infeatures= ... -> 65536 : 입력받을 데이터의 크기(노드의 개수)입니다. Flatten 을 통과한 데이터의 크기와 정확히 일치해야합니다.
            # outfeatures=64 : 이 레이어를 통과한 후 출력될 데이터의 크기입니다. 65536개의 정보를 64개로 압축하는 역할을 합니다.
            nn.Linear(in_features=16 * 64 * 64, out_features=64),
            nn.ReLU(),
            # 이전 레이어에서 64개의 입력을 받아, 우리가 최종적으로 원하는 5개의 숫자(객체 존재 확률 1개, 좌표 4개)를 출력합니다.
            nn.Linear(in_features=64, out_features=5)
        
        def forward(self, x):


        )
        