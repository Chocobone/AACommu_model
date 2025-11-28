from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer # 토크나이저 로딩용 (예시)

# 1. 설정 (학습 때 사용한 모델 구조와 토크나이저가 필요함)
MODEL_PATH = "AACommu_model_best.pt"
TOKENIZER_NAME = "klue/bert-base" # 예시: 학습 때 쓴 토크나이저 이름

app = FastAPI()

# 2. 모델 클래스 정의 (저장된 .pt 파일과 구조가 같아야 함)
# (예시용 가짜 클래스입니다. 실제 사용하시는 모델 클래스를 넣으세요)
class MyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 768)
        self.fc = nn.Linear(768, 30000) # Vocab Size로 출력

    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)

# 3. 모델 및 토크나이저 로드 (서버 시작 시 1회)
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = MyLanguageModel()

# GPU가 있으면 GPU로, 없으면 CPU로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # 추론 모드 필수

# 4. 입력 데이터 정의
class PromptRequest(BaseModel):
    text: str # 예: "오늘 날씨가"

@app.post("/predict/next-token")
def predict_next_token(req: PromptRequest):
    # A. 텍스트 -> 텐서 변환 (Tokenization)
    input_ids = tokenizer.encode(req.text, return_tensors="pt")
    input_ids = input_ids.to(device)

    # B. 추론 (Inference)
    with torch.no_grad():
        outputs = model(input_ids) 
        # outputs 형태는 보통 (Batch_Size, Sequence_Length, Vocab_Size) 입니다.
        
        # 어떤 모델은 outputs가 튜플일 수 있습니다 (예: HuggingFace 모델은 outputs.logits)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

    # C. 다음 토큰 선택 (Next Token Selection)
    # logits[0, -1, :] 의미: 첫번째 배치의, 가장 마지막 단어의, 모든 단어 확률값
    next_token_logits = logits[0, -1, :]
    
    # 가장 확률이 높은 인덱스 찾기 (Argmax)
    next_token_id = torch.argmax(next_token_logits).item()

    # D. 숫자 ID -> 텍스트 변환 (Decoding)
    next_token = tokenizer.decode([next_token_id])

    return {
        "input_text": req.text,
        "next_token": next_token,
        "next_token_id": next_token_id
    }