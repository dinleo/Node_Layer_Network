# Node_Layer_Network
### 개요
![2](https://github.com/dinleo/Node_Layer_Network/assets/81561645/11e17dcd-1fc2-42f2-9663-05fa1e10f759)
- 프레임워크를 사용하지 않고 오직 Numpy로만 MLP, CNN 을 구현하는 프로젝트
    - numpy를 이용해 Dot, Add 등 기본적인 계산그래프의 Node 를 구현
    - → Node 를 이용해 Activation, Affine 등 Layer 를 구현
    - → Layer 를 이용해 FCNMLP, CNN, VGG 등 Network 를 구현
    - → Network 를 용해 학습 및 추론, 테스트

### 참고
- 딥러닝 학습 블로그
    - https://leo-dev.notion.site/b4ad787c62b949b99d3a5b3f2fb4e23b?v=7c6df0ad0b4d40c9b43ac5654dbb15cc&pvs=4
- 교재코드
    - https://github.com/WegraLee/deep-learning-from-scratch
    - '밑바닥부터 시작하는 딥러닝' 교재를 참고해 구현 했습니다.
- 교재와의 차이점
    - 교재
        - Node 가 없습니다.
        - Layer 의 순전파, 역전파는 공식대로 계산합니다.
    - 이 프로젝트
        - Node 는 다른 node 들과는 독립적으로 구현된 계산그래프의 노드 입니다.
        - Node 는 각각 순전파와 역전파가 존재합니다.
        - Layer 는 단순히 Node 를 연결시킨 tree 형태의 자료구조 입니다.
        - Layer 의 순전파, 역전파 함수는 단순히 Node 의 연속적인 순전파 역전파를 실행하기만 합니다.
- 교재와의 공통점
    - 'Networks/' 폴더는 교재와 거의 동일 합니다.

### 구조

- Node 밑 Layer 의 구성은 아래 PPT 에 나와 있습니다.
- [README.pdf](https://github.com/dinleo/Node_Layer_Network/files/12300985/README.pdf)

### 실험

- [Gray white simple modern Thesis Defense Presentation .pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/95332546-857b-48f5-8d93-82a3d7e98eab/Gray_white_simple_modern_Thesis_Defense_Presentation_.pdf)
- 구현한 모델로 교재에 있는 중요한 실험에 대해 동일하게 진행
    - 같은 결과가 나온 실험
        - Optimizer Test
        - Dropout Test
        - Weight Decay
    - 다른 결과가 나온 실험
        - Batch Norm
            - 교재에서는 soft_loss 역전파시 Y-T 사용
            - 이에 순전파시에는 Log(x + 1e-7) 역전파시에는 Log(x) 를 사용한 꼴이 됨
            - 이로인해 생긴 오차
        - 모든 Model 의 Gradient
            - 분수 계산방식 차이에 따른 부동소수점 반올림에 따른 오차
    - Network 실험
        - 데이터셋: mnist
        - 조건: Cupy( GPU ), 128batch acc 0.98 도달
        - 모델: MLP, CNN, VGG
### 업데이트

- PPT 에 없는 구성 추가
    - Layer
        - Batch Norm Layer 추가
        - DropOut Layer 추가
    - Node
        - Power 추가
- CNN 추가
    - 기존 deep_convnet 과 동일, Layer 유연하게 변경할 수 있도록 코드 변경
- VGG16 추가
- Cupy 사용

### 블로그

- https://leo-dev.notion.site/Node_Layer_Network-460c7f529f37465f801563064bfdb683?pvs=4