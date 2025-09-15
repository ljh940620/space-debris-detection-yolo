import torch
import torch.nn as nn
# 오늘 만들 모델 MyMiniDetector는 크게 두 부분으로 나뉩니다.
# Backbone (백본) : 이미지를 입력받아 합성곱 신경망(CNN)을 통해 특징(feature)을 추출합니다. 이미지에 '무엇이 있는지' 이해하는 부분입니다.
# Head (헤드) : 백본이 추출한 특징을 바탕으로, 우리가 원하는 값인 '객체 존재 확률과 위치 좌표' 5개의 숫자를 예측하는 부분입니다.
# 1단계: 모델의 뼈대 만들기(nn.Module 상속)
""" Pytorch의 모든 모델은 nn.Module이라는 기본 클래스를 상속받아 만들어집니다. 이 클래스 안에는 모델을 구성하는
두개의 필수 메소드가 있습니다.
1. __init__(self) : 모델 생성자. 모델에 필요한 모든 부품(레이어)들을 정의하고 초기화하는 곳입니다. 마치 레고를 조립하기 전에
필요한 모든 브릭을 종류별로 꺼내놓는 과정과 같습니다.
2. forward(self, x) : 데이터의 흐름을 정의하는 곳. 입력 데이터 x가 __init__에서 정의한 부품들을
어떤 순서로 통과해서 최종 결과물로 만들어지는지 조립 설명서를 작성하는 과정입니다. """

class MyMiniDetector(nn.Module):
    def __init__(self):
        super(MyMiniDetector, self).__init__()
        # 여기에 모델의 부품(레이어)들을 정의합니다.
        # __init__ 함수 내부
        # 2단계: 백본(Backbone)에서 설계하기: 이미지 '이해'하기
        """ 이제 __init__ 함수 안에 백본을 구성하는 CNN 레이어들을 정의합니다. (합성곱 -> 활성화 함수 -> 풀링) 구조를
        2번  반복하는 간단한 백본을 만들어 보겠습니다.

        데이터 크기 추적의 중요성:
        CNN에서는 레이어를 통과할 때마다 데이터(텐서)의 모양(shape)이 바뀝니다. 이 모양을 계속 추적하는 것이 매우 중요합니다.
        우리의 입력 이미지 크기가 (배치 크기, 채널, 높이, 너비) = (B, 1, 256, 256)라고 가정하고
        크기 변화를 주석으로 달아보겠습니다.

        nn.Sequential : 여러 레이어를 순서대로 묶어주는 편리한 컨테이너입니다. 이렇게 묶어두면 forward 에서 한 번에 호출할 수 있습니다.
        nn.Conv2d : 합성곱 레이어. in_channels는 입력 데이터의 채널 수(흑백=1), out_channels는 특징을 추출할 필터의 개수입니다.
        nn.ReLU : 활성화 함수. 모델에 비선형성을 추가하여 더 복잡한 패턴을 학습하게 합니다.
        nn.MaxPool2d: 맥스 풀링 레이어. 특징 맵의 크기를 줄여 중요한 정보만 남기고 계산량을 감소시킵니다. stride=2 로 설정하면 크기가 절반이 됩니다. """
        # --- 백본(Backbone) 정의 ---
        self.backbone = nn.Sequential(
            # --- Block 1 ---
            # 입력: (B, 1, 256, 256)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            # 출력: (B, 8, 256, 256) - padding=1 덕분에 크기 유지
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 출력: (B, 8, 128, 128) - 높이와 너비가 절반으로 줄어듦

            # --- Block 2 ---
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            # 출력: (B, 16, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 최종 출력: (B, 16, 64, 64)
        )       
        # 3단계: 헤드(Head) 설계하기: '예측'하기
        """ 백본을 통과한 최종 특징 맵 (B, 16, 64, 64) 은 여전히 3차원 데이터입니다. 우리가 원하는 5개의 숫자를 만들기 위해,
        이 데이터를 1차원으로 길게 펼친 뒤 선형 레이어(완전 연결 계층)를 통과시켜야 합니다.

        nn.Flatten() : 다차원 데이터를 1차원으로 쭉 펼쳐줍니다.
        nn.Linear : 선형 레이어. in_features 는 입력 데이터의 크기, out_features 는 출력 데이터의 크기입니다.
        백본의 최종 출력 크기(16 * 64 * 64)를 정확하게 계산해서 in_features 에 넣어주는 것이 매우 중요합니다.  """
        # --- 헤드(Head) 정의 ---
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 64 * 64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=5)
            # 최종 출력: (B, 5)
        )
    # 4단계: 데이터의 흐름 완성하기 (forward 메소드)
    """     이제 __init__에서 정의한 backbone 과 head 를 forward 메소드에서 순서대로 조립합니다.
    torch.sigmoid() : 어떤 값이든 0과 1 사이의 값으로 바꿔주는 함수입니다. 확률을 나타내기에 적합합니다.
    torch.cat([...], dim=1) : 분리했던 텐서들을 다시 하나로 합칩니다. """
    def forward(self, x):
        # 여기에서 데이터가 레이어들을 통과하는 과정을 정의합니다.
        # 1. 이미지를 백본에 통과시켜 특징 추출
        #print(f"입력 shape: {x.shape}")
        features = self.backbone(x)
        #print(f"백본 통과 후 shape: {features.shape}")

        # 2. 추출된 특징을 헤드에 통과시켜 최종 5개 숫자 예측
        output = self.head(features)
        #print(f"헤드 통과 후 shape: {output.shape}")

        # 3. 출력값의 의미에 맞게 후처리 (Activation)
        # 객체 존재 확률(첫 번째 값)은 0~1 사이여야 하므로 Sigmoid 함수 적용
        confidence_score = torch.sigmoid(output[:, 0])

        # 바운딩 박스 좌표(나머지 네 값)는 그대로 사용
        bounding_box = output[:, 1:]

        # 두 결과를 합쳐서 최종 출력 생성
        final_output = torch.cat([confidence_score.unsqueeze(1), bounding_box], dim=1)
        
        return final_output

# 5단계: 전체 코드 및 검증
# 이 모델이 우리가 의도한 대로 잘 작동하는지 확인하는 간단한 검증 코드입니다.
# --- 모델 검증 ---
if __name__ == '__main__':
    # 모델 인스턴스 생성
    model = MyMiniDetector()
    print("모델 구조:")
    print(model)

    # 더미 입력 데이터 생성 (배치 크기=4, 채널=1, 높이=256, 너비=256)
    dummy_input = torch.randn(4, 1, 256, 256)

    # 모델에 데이터 통과
    output = model(dummy_input)

    # 결과 확인
    print("\n--- 검증 ---")
    print(f"입력 텐서 모양: {dummy_input.shape}")
    print(f"출력 텐서 모양: {output.shape}")
    print(f"첫 번째 샘플의 출력값: {output[0]}")




