# Final-Project-II

---

# Tiny GPT: PyTorch 기반 트랜스포머 토이 모델

이 프로젝트는 셰익스피어의 텍스트 데이터를 학습하여 새로운 텍스트를 생성하는 **Tiny GPT (토이 모델)** 의 PyTorch 구현체입니다. 이 모델은 오리지널 트랜스포머(Transformer) 아키텍처 중 텍스트 생성에 특화된 **디코더(Decoder) 전용 구조**를 채택하고 있습니다.

## 🏗️ 전체 아키텍처 (Architecture Overview)

Tiny GPT 모델은 텍스트의 다음 글자(Token)를 예측하는 언어 모델(Language Model)입니다. 모델의 전체적인 입력과 출력 흐름은 다음과 같습니다.

1. **입력 데이터**: 시퀀스 형태의 텍스트 토큰 (크기: `batch_size` x `block_size`)
2. **임베딩 (Embedding)**: 토큰 임베딩 + 위치 임베딩
3. **트랜스포머 블록 (Transformer Blocks)**: N개의 블록을 거치며 문맥(Context)을 파악 (Masked Self-Attention + Feed Forward)
4. **출력 (Output)**: 최종 Layer Normalization 후 Linear 레이어를 거쳐 다음 토큰에 대한 확률 분포(Logits) 출력

---

## 🧩 주요 구성 요소 (Components)

### 1. 임베딩 (Token & Positional Embedding)

모델이 문자를 이해할 수 있도록 숫자로 변환(Tokenization)한 뒤, 이를 고정된 차원의 벡터로 변환합니다.

* **Token Embedding (`token_embedding`)**: 각 문자의 고유한 의미를 벡터 공간에 맵핑합니다.
* **Positional Embedding (`position_embedding`)**: 트랜스포머는 데이터를 한 번에 병렬 처리하므로 단어의 '순서'를 알 수 없습니다. 이를 해결하기 위해 각 위치(Index) 정보도 임베딩하여 토큰 임베딩에 더해줍니다(`h = tok + pos`). 오리지널 논문의 복잡한 주기함수(Sine/Cosine) 대신, 본 모델에서는 학습 가능한(Learnable) 위치 임베딩을 사용합니다.

### 2. 마스크드 멀티-헤드 어텐션 (Masked Multi-Head Attention)

디코더 구조의 핵심으로, 현재 위치의 단어가 '이전 단어들'에게만 주의(Attention)를 기울이도록 합니다.

* **Scaled Dot-Product Attention**:
각 토큰은 **Query(질의), Key(키), Value(값)** 라는 세 가지 벡터로 변환됩니다. Query와 Key의 내적(Dot Product)을 통해 두 토큰 간의 연관성(유사도)을 구하고, 이를 Key 차원의 제곱근(`k.size(-1)  -0.5`)으로 나누어 스케일링합니다.
* **Masking (`tril`)**:
텍스트 생성 모델은 미래의 단어를 보고 현재 단어를 예측하면 안 됩니다 (이를 컨닝 방지라고 생각할 수 있습니다). 하삼각행렬(`torch.tril`)을 사용하여 미래 시점의 연관성 값을 `-inf`로 덮어씌웁니다. 이후 Softmax를 통과하면 미래 토큰에 대한 가중치는 `0`이 됩니다.
* **Multi-Head Attention (`MultiHeadAttention` 클래스)**:
어텐션 연산을 한 번만 하는 것이 아니라, 여러 개의 헤드(Head)로 나누어 병렬로 수행합니다. 각 헤드는 문장의 서로 다른 문맥적 관점(예: 주어-동사 관계, 감정 상태 등)을 포착한 뒤, 다시 하나로 병합(`torch.cat`)합니다.

### 3. 피드포워드 네트워크 (FeedForward Network)

어텐션 레이어가 토큰 간의 '관계'를 파악했다면, 피드포워드 레이어는 파악된 정보를 바탕으로 각 토큰 내부의 특징을 깊이 있게 학습합니다.

* 두 개의 Linear 레이어와 그 사이의 비선형 활성화 함수(`ReLU`)로 구성됩니다.
* 내부적으로 차원을 4배로 늘렸다가 다시 원래 차원으로 줄이면서(`emb_dim -> 4 * emb_dim -> emb_dim`) 모델의 표현력(Capacity)을 극대화합니다.

### 4. 트랜스포머 블록과 잔차 연결 (Transformer Block & Add & Norm)

위에서 설명한 Attention과 FeedForward를 하나로 묶은 단위가 `Block`입니다.

* **Residual Connection (잔차 연결, `x + ...`)**:
입력값을 레이어의 출력값에 그대로 더해줍니다. 이는 딥러닝에서 층이 깊어질수록 발생할 수 있는 기울기 소실(Gradient Vanishing) 문제를 방지합니다.
* **Layer Normalization (`LayerNorm`)**:
학습을 안정화하기 위해 정규화를 수행합니다.
*(💡 아키텍처 팁: 초기 트랜스포머 모델은 Add 연산 이후에 정규화를 하는 Post-LN 구조를 썼지만, 본 GPT 모델은 최근 트렌드에 맞게 Attention과 FeedForward 연산 전에 정규화를 수행하는 **Pre-LN** 구조를 사용하고 있습니다.)*

---

## 🚀 학습 및 텍스트 생성 (Training & Sampling)

* **학습 (Training)**:
정답이 되는 다음 텍스트(Target)와 모델의 예측값(Logits) 간의 차이를 `CrossEntropy` 손실 함수로 계산하여, AdamW 옵티마이저를 통해 모델의 가중치를 업데이트합니다.
* **샘플링 (Sampling)**:
학습된 모델에 초기 텍스트(예: `"ROMEO:"`)를 주면, 모델은 문맥을 파악하여 다음 글자의 확률 분포를 계산합니다. `torch.multinomial`을 사용하여 확률에 기반한 난수 추출 방식으로 다음 글자를 생성하고, 이를 다시 입력으로 넣어 반복적으로 텍스트를 자동 생성합니다.
