깃헙(GitHub) 환경이나 로컬 저장소에서 코드를 원활하게 실행할 수 있도록, 기존 구글 코랩(Colab)의 절대 경로(`/content/...`)로 하드코딩되어 있던 파일 경로를 동일한 디렉토리에 있는 파일을 읽을 수 있게 상대 경로로 변경했습니다. 또한, 예외 처리 메시지도 코랩 환경뿐만 아니라 일반적인 환경에 맞게 수정했습니다.

아래는 변경이 완료된 주피터 노트북(Jupyter Notebook) JSON 코드입니다. 이 코드를 복사하여 `.ipynb` 파일에 덮어쓰시면 됩니다.

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "---\n",
        "\n",
        "### **1. 데이터 로드 및 전처리 (Data Loading & Preprocessing)**\n",
        "\n",
        "새로운 텍스트 파일로 어휘집(vocabulary)을 만들고, 모델이 학습할 수 있도록 슬라이딩 윈도우 방식의 데이터셋을 정의합니다."
      ],
      "metadata": {
        "id": "P269mmqsqH6S"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch # PyTorch 메인 라이브러리 임포트\n",
        "import torch.nn as nn # 신경망 구축을 위한 모듈(레이어 등) 임포트\n",
        "import torch.nn.functional as F # 활성화 함수, 손실 함수 등 함수형 API 임포트\n",
        "from torch.utils.data import Dataset, DataLoader # 커스텀 데이터셋 및 배치 생성을 위한 모듈 임포트\n",
        "from pathlib import Path # 파일 및 디렉토리 경로를 객체로 다루기 위한 라이브러리 임포트\n",
        "\n",
        "# 학습에 사용할 텍스트 파일 경로 지정 (GitHub 레포지토리 내 상대 경로 사용)\n",
        "file_path = \"Todd G. Buchholz - New Ideas from Dead Economists_ The Introduction to Modern Economic Thought, 4th Edition (2021, Penguin Publishing Group).txt\"\n",
        "\n",
        "# 텍스트 파일 읽기 시도\n",
        "try:\n",
        "    with open(file_path, \"r\", encoding=\"utf-8\") as f: # 파일을 읽기 모드('r')와 UTF-8 인코딩으로 열기\n",
        "        text_eco = f.read() # 파일 내의 모든 텍스트를 하나의 문자열로 읽어오기\n",
        "\n",
        "    print(f\"File length: {len(text_eco)} characters\") # 로드된 텍스트의 총 문자 수 출력\n",
        "\n",
        "    # 문자 단위 어휘집(Character-level vocabulary) 생성\n",
        "    chars_eco = sorted(list(set(text_eco))) # 텍스트 내의 고유한 문자들만 추출하여 오름차순으로 정렬 (Vocabulary 생성)\n",
        "    vocab_size_eco = len(chars_eco) # 고유 문자의 총 개수를 어휘집 크기(vocab_size)로 설정\n",
        "    print(f\"Unique characters: {vocab_size_eco}\") # 어휘집의 크기 출력\n",
        "\n",
        "    # 문자와 정수 간의 매핑(Mappings) 딕셔너리 생성\n",
        "    stoi_eco = {ch: i for i, ch in enumerate(chars_eco)} # 문자(ch)를 정수 인덱스(i)로 변환하는 딕셔너리 생성 (String to Integer)\n",
        "    itos_eco = {i: ch for ch, i in stoi_eco.items()} # 정수 인덱스(i)를 문자(ch)로 변환하는 딕셔너리 생성 (Integer to String)\n",
        "\n",
        "    # 텍스트 데이터를 PyTorch 텐서로 변환\n",
        "    # Size Conformity: text_eco의 길이가 N일 때, [N] 형태의 1차원 정수 텐서 생성\n",
        "    data_eco = torch.tensor([stoi_eco[ch] for ch in text_eco], dtype=torch.long)\n",
        "    print(f\"Data tensor shape: {data_eco.shape}\") # 생성된 텐서의 형태 출력\n",
        "\n",
        "except FileNotFoundError: # 지정된 경로에 파일이 없을 경우의 예외 처리\n",
        "    print(f\"Error: The file was not found at {file_path}. 동일한 디렉토리에 텍스트 파일이 존재하는지 확인해주세요.\")\n",
        "\n",
        "# 다음 토큰 예측을 위한 커스텀 데이터셋 클래스 정의 (슬라이딩 컨텍스트 윈도우 적용)\n",
        "class NextTokenDataset(Dataset):\n",
        "    def __init__(self, data, block_size): # 초기화 메서드\n",
        "        self.data = data # 전체 텍스트가 정수로 변환된 1차원 텐서 저장\n",
        "        self.block_size = block_size # 모델이 한 번에 입력으로 받을 문맥 윈도우의 크기 (T) 설정\n",
        "\n",
        "    def __len__(self): # 데이터셋의 전체 길이 반환\n",
        "        return len(self.data) - self.block_size # 마지막 block_size 만큼은 타겟을 만들 수 없으므로 제외\n",
        "\n",
        "    def __getitem__(self, idx): # 특정 인덱스(idx)에 해당하는 데이터 샘플 반환\n",
        "        # Size Conformity: x와 y 모두 길이는 block_size. 즉, 형태는 [T]\n",
        "        x = self.data[idx : idx + self.block_size] # 입력 시퀀스: idx부터 block_size 길이만큼 추출\n",
        "        y = self.data[idx + 1 : idx + self.block_size + 1] # 타겟 시퀀스: x보다 1스텝 뒤로 밀린 시퀀스 추출\n",
        "        return x, y # 입력과 타겟 텐서를 튜플로 반환\n",
        "\n",
        "block_size = 64 # 컨텍스트 윈도우 크기를 64로 설정 (T = 64)\n",
        "dataset = NextTokenDataset(data_eco, block_size) # 준비된 텐서와 block_size로 데이터셋 인스턴스 생성\n",
        "# 미니 배치를 생성하는 DataLoader 설정\n",
        "# Size Conformity: batch_size=64이므로, loader에서 나오는 배치는 x: [64, 64], y: [64, 64] 형태가 됨 (B=64, T=64)\n",
        "loader = DataLoader(dataset, batch_size=64, shuffle=True)\n",
        "xb, yb = next(iter(loader)) # 로더에서 첫 번째 배치를 추출하여 형태(shape) 테스트 준비"
      ],
      "outputs": [],
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lWUdSe1pqH6U"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### `chars_eco` 내 비알파벳 및 비숫자 문자 설명\n",
        "\n",
        "추출된 데이터를 바탕으로 텍스트에서 발견된 특수 문자들을 분류하면 다음과 같습니다:\n",
        "\n",
        "1. **공백 및 제어 문자:**\n",
        "    - `\\t`: 탭 문자 (PDF를 텍스트로 변환할 때 들여쓰기용으로 주로 사용됨).\n",
        "    - `\\n`: 줄바꿈 문자 (단락이나 행을 나눔).\n",
        "    - ` `: 일반 공백.\n",
        "\n",
        "2. **표준 문장 부호:**\n",
        "    - `!`, `#`, `$`, `&`, `(`, `)`, `*`, `+`, `,`, `-`, `.`, `/`, `:`, `;`, `=`, `>`, `?`: 문장, 인용, 또는 구분 기호로 사용되는 일반적인 부호들.\n",
        "\n",
        "3. **괄호 및 서식:**\n",
        "    - `[`, `]`: 주로 인용문이나 편집자 주석에 사용됨.\n",
        "    - `_`, `|`: 스타일 지정, 구분선 또는 자리 표시자로 사용됨.\n",
        "\n",
        "4. **따옴표 및 특수 대시:**\n",
        "    - `‘`, `’`, `“`, `”`: 대화나 강조를 위한 스마트(곡선) 따옴표.\n",
        "    - `–` (En-dash), `—` (Em-dash): 범위나 생각의 중단을 나타낼 때 사용됨.\n",
        "    - `−`: 공식적인 마이너스 기호 (하이픈과 구별됨).\n",
        "\n",
        "5. **통화 및 저작권:**\n",
        "    - `£`: 영국 파운드 기호 (경제학 논의에서 자주 등장).\n",
        "    - `©`: 저작권 기호.\n",
        "\n",
        "6. **액센트 문자 (외국어 인명/용어):**\n",
        "    - `Ü`, `à`, `á`, `â`, `ä`, `ç`, `è`, `é`, `ê`, `ò`, `ö`, `ú`, `û`, `ü`, `ğ`, `ł`, `ń`, `ō`: 경제학자 이름(예: 프랑스어, 독일어, 터키어 이름)이나 책에서 사용된 특정 학술 용어에 포함된 문자들.\n",
        "\n",
        "7. **수학 기호:**\n",
        "    - `×`: 곱셈 기호."
      ],
      "metadata": {
        "id": "rjUm7ad3qg2-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "---\n",
        "\n",
        "### **2. 멀티 헤드 어텐션 (Multi-head attention)**\n",
        "\n",
        "Self-Attention 메커니즘을 구현합니다. 이 단계에서는 쿼리(Query), 키(Key), 밸류(Value) 행렬 간의 연산에서 **Size Conformity**가 모델 성능의 핵심이 됩니다.\n",
        "\n",
        "수학적으로 어텐션 스코어는 다음과 같이 계산됩니다:\n",
        "$Attention(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$"
      ],
      "metadata": {
        "id": "QyJ9Z5DhqH6W"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class Head(nn.Module): # 단일 어텐션 헤드 클래스 정의\n",
        "    def __init__(self, emb_dim, head_size, block_size, dropout=0.1): # 초기화 메서드\n",
        "        super().__init__() # 부모 클래스(nn.Module) 초기화\n",
        "        # 선형 변환 레이어 정의 (편향 bias=False 사용)\n",
        "        self.key = nn.Linear(emb_dim, head_size, bias=False) # 임베딩 차원을 head_size로 변환하는 Key 레이어\n",
        "        self.query = nn.Linear(emb_dim, head_size, bias=False) # 임베딩 차원을 head_size로 변환하는 Query 레이어\n",
        "        self.value = nn.Linear(emb_dim, head_size, bias=False) # 임베딩 차원을 head_size로 변환하는 Value 레이어\n",
        "        # 삼각 행렬 버퍼 등록 (미래 토큰을 참조하지 못하게 하는 마스킹 용도, 모델 파라미터가 아님)\n",
        "        self.register_buffer(\"tril\", torch.tril(torch.ones(block_size, block_size)))\n",
        "        self.dropout = nn.Dropout(dropout) # 과적합 방지를 위한 드롭아웃 레이어\n",
        "\n",
        "    def forward(self, x): # 순전파 메서드\n",
        "        # x 입력 형태: [B(배치 크기), T(시퀀스 길이), C(임베딩 차원 emb_dim)]\n",
        "        B, T, C = x.shape\n",
        "\n",
        "        # Size Conformity: x [B, T, C] -> 선형 변환 -> [B, T, head_size]\n",
        "        k = self.key(x) # Key 벡터 생성: 크기는 [B, T, head_size]\n",
        "        q = self.query(x) # Query 벡터 생성: 크기는 [B, T, head_size]\n",
        "        v = self.value(x) # Value 벡터 생성: 크기는 [B, T, head_size]\n",
        "\n",
        "        # 어텐션 스코어 계산 (Query 와 Key의 내적)\n",
        "        # Size Conformity: q [B, T, head_size] @ k의 전치 [B, head_size, T] = wei [B, T, T]\n",
        "        # (k.size(-1) ** -0.5)를 곱해 스케일링(분산 정규화) 수행\n",
        "        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)\n",
        "\n",
        "        # 마스킹 적용: 상삼각행렬 부분을 -inf로 채워 미래 시점의 정보 차단\n",
        "        # Size Conformity: wei [B, T, T] 형태 유지\n",
        "        wei = wei.masked_fill(self.tril[:T, :T] == 0, float(\"-inf\"))\n",
        "\n",
        "        # 소프트맥스 적용: 마지막 차원(dim=-1)을 기준으로 확률 분포로 변환 (합이 1)\n",
        "        wei = F.softmax(wei, dim=-1) # wei 형태: [B, T, T]\n",
        "        wei = self.dropout(wei) # 어텐션 가중치에 드롭아웃 적용\n",
        "\n",
        "        # 어텐션 스코어와 Value 행렬 곱셈\n",
        "        # Size Conformity: wei [B, T, T] @ v [B, T, head_size] = out [B, T, head_size]\n",
        "        out = wei @ v\n",
        "        return out # 최종 결과 반환 [B, T, head_size]\n",
        "\n",
        "class MultiHeadAttention(nn.Module): # 멀티 헤드 어텐션 클래스 정의\n",
        "    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1): # 초기화 메서드\n",
        "        super().__init__() # 부모 클래스 초기화\n",
        "        head_size = emb_dim // num_heads # 각 헤드가 담당할 차원 수 계산 (예: 128 // 4 = 32)\n",
        "        # 여러 개의 단일 헤드를 리스트 형태로 구성 (num_heads 개수만큼)\n",
        "        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])\n",
        "        # 결과를 원래 임베딩 차원으로 복구/투영하는 선형 레이어\n",
        "        self.proj = nn.Linear(emb_dim, emb_dim)\n",
        "        self.dropout = nn.Dropout(dropout) # 드롭아웃 레이어\n",
        "\n",
        "    def forward(self, x): # 순전파 메서드\n",
        "        # 각 헤드의 결과물들을 마지막 차원(dim=-1) 기준으로 연결(concatenate)\n",
        "        # Size Conformity: 개별 헤드 출력 [B, T, head_size]가 num_heads개 모임 -> [B, T, head_size * num_heads] = [B, T, emb_dim]\n",
        "        out = torch.cat([h(x) for h in self.heads], dim=-1)\n",
        "        out = self.proj(out) # 투영 레이어 통과 (형태 유지: [B, T, emb_dim])\n",
        "        out = self.dropout(out) # 드롭아웃 적용\n",
        "        return out # 최종 멀티 헤드 어텐션 결과 반환"
      ],
      "outputs": [],
      "execution_count": null,
      "metadata": {
        "id": "-vxdAqTWqH6X"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "---\n",
        "\n",
        "### **3. 피드포워드 네트워크 및 블록 구조 (Feedforward + Block)**\n",
        "\n",
        "Transformer 아키텍처의 개별 Block(어텐션 + 피드포워드 + 정규화 + 잔차 연결)을 구성합니다."
      ],
      "metadata": {
        "id": "TqOgf6u8qH6X"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class FeedForward(nn.Module): # 위치 단위(position-wise) 피드포워드 네트워크 클래스\n",
        "    def __init__(self, emb_dim, dropout=0.1): # 초기화 메서드\n",
        "        super().__init__() # 부모 클래스 초기화\n",
        "        self.net = nn.Sequential( # 여러 레이어를 순차적으로 묶음\n",
        "            # Size Conformity: [B, T, emb_dim] -> [B, T, 4 * emb_dim] (내부적으로 차원 확장)\n",
        "            nn.Linear(emb_dim, 4 * emb_dim),\n",
        "            nn.ReLU(), # 비선형 활성화 함수 ReLU 적용\n",
        "            # Size Conformity: [B, T, 4 * emb_dim] -> [B, T, emb_dim] (원래 차원으로 축소)\n",
        "            nn.Linear(4 * emb_dim, emb_dim),\n",
        "            nn.Dropout(dropout), # 드롭아웃 적용\n",
        "        )\n",
        "    def forward(self, x): # 순전파 메서드\n",
        "        return self.net(x) # x 입력 후 형태 변화 없이 반환 [B, T, emb_dim]\n",
        "\n",
        "class Block(nn.Module): # 트랜스포머 블록 클래스 (Attention + FeedForward)\n",
        "    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1): # 초기화 메서드\n",
        "        super().__init__() # 부모 클래스 초기화\n",
        "        self.ln1 = nn.LayerNorm(emb_dim) # 첫 번째 레이어 정규화 (Layer Normalization)\n",
        "        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout) # 멀티 헤드 어텐션 레이어\n",
        "        self.ln2 = nn.LayerNorm(emb_dim) # 두 번째 레이어 정규화\n",
        "        self.ffwd = FeedForward(emb_dim, dropout) # 피드포워드 네트워크 레이어\n",
        "\n",
        "    def forward(self, x): # 순전파 메서드\n",
        "        # Size Conformity: x [B, T, emb_dim] + sa_output [B, T, emb_dim] -> 형태 유지. 잔차 연결(Residual Connection)\n",
        "        x = x + self.sa(self.ln1(x)) # 입력 -> 정규화 -> 어텐션 -> 입력과 더하기\n",
        "        # Size Conformity: x [B, T, emb_dim] + ffwd_output [B, T, emb_dim] -> 형태 유지. 잔차 연결\n",
        "        x = x + self.ffwd(self.ln2(x)) # 입력 -> 정규화 -> 피드포워드 -> 입력과 더하기\n",
        "        return x # 블록 연산을 마친 텐서 반환 [B, T, emb_dim]"
      ],
      "outputs": [],
      "execution_count": null,
      "metadata": {
        "id": "XXDO9_JAqH6Y"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "---\n",
        "\n",
        "### **4. Tiny GPT 아키텍처 모델 조립 (Tiny GPT)**\n",
        "\n",
        "이제 앞서 만든 모듈들을 모두 조립하여 GPT 모델을 완성합니다."
      ],
      "metadata": {
        "id": "kXVTwOqiqH6Y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class TinyGPT(nn.Module): # Tiny GPT 메인 모델 클래스\n",
        "    def __init__(self, vocab_size, block_size, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1): # 초기화 메서드\n",
        "        super().__init__() # 부모 클래스 초기화\n",
        "        # Size Conformity: 토큰 인덱스 입력을 [B, T] -> [B, T, emb_dim] 크기의 밀집 벡터(dense vector)로 임베딩\n",
        "        self.token_embedding = nn.Embedding(vocab_size, emb_dim)\n",
        "        # Size Conformity: 위치 인덱스를 [T] -> [T, emb_dim] 크기의 위치 벡터로 임베딩\n",
        "        self.position_embedding = nn.Embedding(block_size, emb_dim)\n",
        "\n",
        "        # num_layers 개수만큼 Block을 쌓아서 순차적인 네트워크 구성\n",
        "        self.blocks = nn.Sequential(*[\n",
        "            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)\n",
        "        ])\n",
        "        self.ln_f = nn.LayerNorm(emb_dim) # 트랜스포머 블록들을 모두 통과한 후의 최종 레이어 정규화\n",
        "\n",
        "        # Size Conformity: 최종 출력을 위해 임베딩 차원을 어휘집 크기(vocab_size)로 변환 [B, T, emb_dim] -> [B, T, vocab_size]\n",
        "        self.lm_head = nn.Linear(emb_dim, vocab_size)\n",
        "\n",
        "    def forward(self, x): # 순전파 메서드\n",
        "        B, T = x.shape # x의 형태 확인: Batch 크기(B), 시퀀스 길이(T)\n",
        "\n",
        "        # 디바이스(CPU/GPU)에 맞게 0부터 T-1까지의 정수 위치 텐서 생성. 형태: [T]\n",
        "        pos = torch.arange(T, device=x.device)\n",
        "\n",
        "        # 토큰 임베딩 통과\n",
        "        # Size Conformity: [B, T] -> [B, T, emb_dim]\n",
        "        tok = self.token_embedding(x)\n",
        "\n",
        "        # 위치 임베딩 통과 및 차원 확장\n",
        "        # Size Conformity: [T] -> [T, emb_dim] -> [None]을 통해 [1, T, emb_dim]으로 차원 브로드캐스팅 준비\n",
        "        pos = self.position_embedding(pos)[None]\n",
        "\n",
        "        # 토큰 정보와 위치 정보를 더함 (Broadcasting을 통한 Size Conformity 성립)\n",
        "        # [B, T, emb_dim] + [1, T, emb_dim] = [B, T, emb_dim]\n",
        "        h = tok + pos\n",
        "\n",
        "        # 누적된 트랜스포머 블록들을 순차적으로 통과 (형태 변화 없음 [B, T, emb_dim])\n",
        "        h = self.blocks(h)\n",
        "\n",
        "        # 최종 레이어 정규화 적용 (형태 변화 없음 [B, T, emb_dim])\n",
        "        h = self.ln_f(h)\n",
        "\n",
        "        # 최종 언어 모델 헤드를 통과하여 다음 토큰에 대한 로짓(logits) 계산\n",
        "        # Size Conformity: [B, T, emb_dim] -> 선형 투영 -> [B, T, vocab_size]\n",
        "        logits = self.lm_head(h)\n",
        "        return logits # 계산된 로짓 반환\n",
        "\n",
        "# 모델 초기화 및 형태 테스트\n",
        "model = TinyGPT(vocab_size_eco, block_size) # 새로운 vocab_size_eco를 사용하여 모델 생성\n",
        "logits = model(xb) # 첫 번째 배치(xb)를 통과시켜 예측 수행\n",
        "print(\"logits.shape:\", logits.shape) # 형태 출력: [64, 64, vocab_size_eco]가 나와야 함"
      ],
      "outputs": [],
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9inea23pqH6Z"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "---\n",
        "\n",
        "### **5. 학습 (Training Loop)**"
      ],
      "metadata": {
        "id": "8mJfwdt0qH6Z"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def sequence_cross_entropy(logits, targets): # 시퀀스 데이터에 대한 크로스 엔트로피 손실 함수 계산\n",
        "    # CrossEntropyLoss는 입력으로 [B, 클래스 수, T] 형태를 기대하므로 차원 전치가 필요함\n",
        "    # Size Conformity: logits [B, T, vocab_size] -> transpose(1, 2) -> [B, vocab_size, T]\n",
        "    # targets 형태: [B, T]. PyTorch 내부적으로 각 위치별 손실을 계산 후 평균 반환\n",
        "    return F.cross_entropy(logits.transpose(1, 2), targets)\n",
        "\n",
        "def train_one_epoch(model, loader, optimizer, device, max_steps=None): # 1 에폭(epoch) 학습 함수\n",
        "    model.train() # 모델을 학습 모드로 설정 (드롭아웃, 배치 정규화 등 활성화)\n",
        "    total_loss, total_count = 0.0, 0 # 누적 손실값과 처리된 데이터 개수 초기화\n",
        "\n",
        "    for step, (xb, yb) in enumerate(loader): # 미니 배치 단위로 데이터 순회\n",
        "        xb, yb = xb.to(device), yb.to(device) # 입력 데이터와 타겟 데이터를 계산할 디바이스(GPU 등)로 이동\n",
        "        logits = model(xb) # 모델 순전파 (예측 로짓 획득)\n",
        "        loss = sequence_cross_entropy(logits, yb) # 예측 로짓과 실제 타겟 간의 손실(Loss) 계산\n",
        "\n",
        "        optimizer.zero_grad() # 이전 스텝의 기울기(Gradient) 초기화\n",
        "        loss.backward() # 역전파(Backpropagation)를 통해 모든 파라미터에 대한 기울기 계산\n",
        "        optimizer.step() # 옵티마이저를 사용하여 파라미터 업데이트\n",
        "\n",
        "        # 현재 배치의 손실을 전체 배치 크기에 곱하여 누적 (평균 손실을 정확히 계산하기 위함)\n",
        "        total_loss += loss.item() * xb.size(0)\n",
        "        total_count += xb.size(0) # 처리된 샘플 수 누적\n",
        "\n",
        "        if max_steps is not None and step + 1 >= max_steps: # 지정된 최대 스텝에 도달하면 학습 중단\n",
        "            break\n",
        "\n",
        "    return total_loss / total_count # 1 에폭의 평균 손실 반환\n",
        "\n",
        "# GPU 사용 가능 여부에 따라 연산 디바이스 설정 (cuda 또는 cpu)\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "# 모델을 생성하고 지정된 디바이스로 이동\n",
        "model = TinyGPT(vocab_size_eco, block_size).to(device)\n",
        "# AdamW 옵티마이저 설정 (학습률 3e-4)\n",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)\n",
        "\n",
        "# 지정된 에폭 수(여기서는 100회)만큼 반복 학습\n",
        "for epoch in range(100):\n",
        "    # 한 에폭을 학습시키고(최대 300스텝) 반환된 평균 손실값을 저장\n",
        "    train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=300)\n",
        "    print(f\"epoch {epoch:2d} | train loss {train_loss:.4f}\") # 현재 에폭과 학습 손실 출력"
      ],
      "outputs": [],
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Hoi-Ek1fqH6a"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "---\n",
        "\n",
        "### **6. 경제학 서적 기반 샘플링 (Sampling \"Milton Friedman\")**\n",
        "\n",
        "학습된 모델에 `Milton Friedman`이라는 프롬프트를 주어 텍스트 생성을 실험합니다."
      ],
      "metadata": {
        "id": "CDdLim0IqH6a"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "@torch.no_grad() # 텍스트 생성 과정에서는 기울기 계산이 불필요하므로 메모리와 연산량 절약을 위해 비활성화\n",
        "def sample_gpt(model, block_size, stoi, itos, device, start_text=\"Milton Friedman\", max_new_tokens=400): # 샘플링 함수 정의\n",
        "    model.eval() # 모델을 평가 모드로 전환 (드롭아웃 비활성화)\n",
        "\n",
        "    # 모델에 입력될 초기 컨텍스트 윈도우를 0으로 채워서 생성\n",
        "    # Size Conformity: [1(배치), block_size(문맥 길이)]\n",
        "    context = torch.zeros((1, block_size), dtype=torch.long, device=device)\n",
        "\n",
        "    # 사용자가 제공한 시작 텍스트(start_text)를 컨텍스트 윈도우에 밀어 넣음\n",
        "    for ch in start_text:\n",
        "        if ch in stoi: # 문자가 어휘집에 존재하는 경우\n",
        "            # Size Conformity: 해당 문자의 정수 인덱스를 [1, 1] 크기의 텐서로 변환\n",
        "            ix = torch.tensor([[stoi[ch]]], device=device)\n",
        "            # 기존 컨텍스트의 가장 앞부분 1칸을 버리고, 맨 뒤에 새 인덱스를 이어붙임 (Sliding Window)\n",
        "            # Size Conformity: context[:, 1:]은 [1, block_size-1] 이고 ix는 [1, 1]. cat(dim=1)을 통해 다시 [1, block_size] 유지\n",
        "            context = torch.cat([context[:, 1:], ix], dim=1)\n",
        "\n",
        "    out = list(start_text) # 생성된 문자를 누적할 리스트 (초기값으로 start_text 삽입)\n",
        "\n",
        "    # 지정된 최대 생성 토큰 수만큼 반복하여 글자 생성\n",
        "    for _ in range(max_new_tokens):\n",
        "        # 현재 컨텍스트를 모델에 통과시킴\n",
        "        # Size Conformity: 입력 [1, block_size] -> 출력 logits [1, block_size, vocab_size]\n",
        "        logits = model(context)\n",
        "\n",
        "        # 다음 문자를 예측하기 위해 마지막 타임스텝(-1)의 출력만 가져옴\n",
        "        # Size Conformity: [1, block_size, vocab_size] -> [1, vocab_size]\n",
        "        logits = logits[:, -1, :]\n",
        "\n",
        "        # 소프트맥스 함수를 적용하여 로짓을 확률 분포로 변환\n",
        "        probs = F.softmax(logits, dim=-1) # 형태 유지: [1, vocab_size]\n",
        "\n",
        "        # 다항 분포(Multinomial)에서 샘플링을 통해 다음 글자의 인덱스 추출 (랜덤성 부여)\n",
        "        # Size Conformity: 1개의 샘플을 뽑으므로 ix 형태는 [1, 1]\n",
        "        ix = torch.multinomial(probs, num_samples=1)\n",
        "\n",
        "        # 뽑힌 정수 인덱스를 다시 문자로 변환하여 결과 리스트에 추가\n",
        "        out.append(itos[ix.item()])\n",
        "\n",
        "        # 방금 생성한 글자를 컨텍스트의 마지막에 추가하여 윈도우를 한 칸 슬라이딩\n",
        "        # Size Conformity: [1, block_size-1]과 [1, 1] 결합 -> [1, block_size] 유지\n",
        "        context = torch.cat([context[:, 1:], ix], dim=1)\n",
        "\n",
        "    return \"\".join(out) # 리스트에 담긴 글자들을 하나의 문자열로 합쳐서 반환\n",
        "\n",
        "# \"Milton Friedman\"을 시작 문구로 하여 500자의 텍스트를 샘플링하고 결과 출력\n",
        "print(sample_gpt(model, block_size, stoi_eco, itos_eco, device, start_text=\"Monetarist\", max_new_tokens=1000))"
      ],
      "outputs": [],
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5uFvTHQ7qH6b"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "zrWN35DqwZbN"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

```
